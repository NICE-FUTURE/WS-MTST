import time
import logging
import traceback
import argparse
import os
from datetime import datetime

from omegaconf import OmegaConf
from thop import profile

import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F

from datasets import BraTSDataset
from utils import (
    AverageMeter,
    BraTSTransform,
    ListMeter,
    WarmupCosineSchedule,
    cal_eta,
    dice_coef,
    fix_seed,
    plot_history,
    multi_label_accuracy,
)
from network.model import Model, AVAILABLE_MODELS
from losses import aggregation_loss, overlapping_loss, FocalLoss


def main():
    ### model
    if args.weights:
        if args.weights == "imagenet":
            default_pretrain = True
            logging.warning("Loading default ImageNet pretrain weights.")
        else:
            default_pretrain = False
            logging.warning(
                "Loading self-supervised pretrain weights:{}.".format(args.weights)
            )
    else:
        default_pretrain = False
        logging.warning("Loading NO pretrain weights.")
    model = Model(
        backbone=args.arch,
        num_classes=cfg.dataset.num_classes,
        decoder_dim=256,
        pretrained=default_pretrain,
        num_channels=cfg.dataset.num_channels,
        out_size=cfg.dataset.img_size,
    )

    ### optimizer
    backbone_params, _, cls_params, decode_params = model.get_param_groups()
    optimizers = [
        torch.optim.AdamW(
            backbone_params,
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
            betas=[0.9, 0.999],
        ),
        torch.optim.AdamW(
            cls_params,
            lr=cfg.optimizer.learning_rate * 10,
            weight_decay=cfg.optimizer.weight_decay,
            betas=[0.9, 0.999],
        ),
        torch.optim.AdamW(
            decode_params,
            lr=cfg.optimizer.learning_rate * 10,
            weight_decay=cfg.optimizer.weight_decay,
            betas=[0.9, 0.999],
        ),
    ]

    ### load self-supervised weights
    if not args.resume and (args.weights and args.weights != "imagenet"):
        state_dict = torch.load(args.weights, map_location=device)
        backone_embedding_params = []
        for key in list(state_dict.keys()):
            value = state_dict.pop(key)
            if key.startswith("backbone."):
                state_dict[key.replace("backbone.", "")] = value
            if key.startswith("backbone_embedding"):
                backone_embedding_params.append((key, value))
        if len(backone_embedding_params) > 0:
            logging.info("update the weights of the patch_embed1 in backbone.")
            for key, value in backone_embedding_params:
                state_dict[key.replace("backbone_embedding", "patch_embed1")] = value
        model.encoder.load_state_dict(state_dict)
        logging.info("Self-supervised pretrain weights loaded.")
    ### load resume weights
    best_val = None
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume + "last.pth", map_location=device)
        checkpoint = {
            k: v
            for k, v in checkpoint.items()
            if not (k.endswith("total_ops") or k.endswith("total_params"))
        }
        model.load_state_dict(checkpoint)
        checkpoint = torch.load(model_dir + "params.pth")
        [
            optimizers[idx].load_state_dict(dict)
            for idx, dict in enumerate(checkpoint["optimizer_state_dicts"])
        ]
        start_epoch = checkpoint["epoch"]
        best_val = checkpoint["val_loss"]
        logging.info("Resuming params loaded.")

    model.to(device)

    logging.info("\nargs: {}".format(args))
    logging.info("\nconfigs: {}".format(cfg))
    logging.info("\nNetwork config: \n{}".format(model))
    logging.info("Calculate MACs & FLOPs ...")
    inputs = torch.randn((1, 4, cfg.dataset.img_size, cfg.dataset.img_size)).to(device)
    macs, num_params = profile(model, (inputs,), verbose=False)  # type: ignore
    logging.info(
        "\nParams(M):{:.2f}, MACs(G):{:.2f}, FLOPs(G):~{:.2f}".format(
            num_params / (1000**2), macs / (1000**3), 2 * macs / (1000**3)
        )
    )
    logging.info("")

    ### transform
    train_transform = BraTSTransform(image_size=cfg.dataset.img_size)
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    ### dataset & dataloader
    train_dataset = BraTSDataset(
        txt_dir=cfg.dataset.txt_dir,
        root_dir=cfg.dataset.root_dir,
        stage=cfg.train.stage,
        transform=train_transform,
    )
    val_dataset = BraTSDataset(
        txt_dir=cfg.dataset.txt_dir,
        root_dir=cfg.dataset.root_dir,
        stage=cfg.val.stage,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.val.batch_size,
        shuffle=False,
        num_workers=cfg.val.num_workers,
        pin_memory=True,
    )

    schedulers = [
        WarmupCosineSchedule(
            optimizer,
            warmup_steps=cfg.optimizer.warmup_epochs,
            t_total=int(1.1 * cfg.train.epochs),
        )
        for optimizer in optimizers
    ]
    decode_criterion = FocalLoss(
        num_class=cfg.dataset.num_classes, weights=None, ignore_index=0
    )

    logging.info("START TIME:{}".format(time.asctime(time.localtime(time.time()))))
    start_time = datetime.now().replace(microsecond=0)
    list_meter = ListMeter()
    for epoch in range(start_epoch, cfg.train.epochs):
        # train
        loss, acc, recall, dice = train(
            train_loader, model, optimizers, epoch, decode_criterion
        )
        [scheduler.step() for scheduler in schedulers]
        list_meter.add({"loss": loss, "acc": acc, "recall": recall, "dice": dice})
        # validate
        val_loss, val_acc, val_recall, val_dice = validate(
            val_loader, model, epoch, decode_criterion
        )
        list_meter.add(
            {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_recall": val_recall,
                "val_dice": val_dice,
            }
        )

        eta = cal_eta(start_time, epoch + 1, cfg.train.epochs)
        logging.info(
            "epoch:{:<5}/{:<5} ".format(epoch + 1, cfg.train.epochs)
            + "eta:{} ".format(eta)
            + "lr:{:.6f} ".format(optimizers[0].param_groups[0]["lr"])
            + "loss:{:.6f} val_loss:{:.6f} ".format(loss, val_loss)
            + "acc:{:.6f} val_acc:{:.6f} ".format(acc, val_acc)
            + "recall:{:.6f} val_recall:{:.6f} ".format(recall, val_recall)
            + "dice:{:.6f} val_dice:{:.6f} ".format(dice, val_dice)
        )
        plot_history(
            list_meter.get(
                "loss",
                "acc",
                "recall",
                "dice",
                "val_loss",
                "val_acc",
                "val_recall",
                "val_dice",
            ),
            history_dir,
        )

        # save model
        torch.save(
            {
                "epoch": epoch + 1,
                "optimizer_state_dicts": [
                    optimizer.state_dict() for optimizer in optimizers
                ],
                "val_loss": val_loss,
            },
            model_dir + "params.pth",
        )
        torch.save(model.state_dict(), os.path.join(model_dir + "last.pth"))
        if best_val is None or val_loss < best_val:  # type: ignore
            best_val = val_loss
            torch.save(model.state_dict(), model_dir + "best.pth")
            logging.info("Saved best model.")

    plot_history(
        list_meter.get(
            "loss",
            "acc",
            "recall",
            "dice",
            "val_loss",
            "val_acc",
            "val_recall",
            "val_dice",
        ),
        history_dir,
    )
    logging.info("STOP TIME:{}".format(time.asctime(time.localtime(time.time()))))


def train(train_loader, model, optimizers, epoch, decode_criterion):
    model.train()
    avg_meter = AverageMeter()
    total = len(train_loader)
    for i, (images, cls_labels, masks, infos) in enumerate(train_loader):
        images = images.to(device)
        cls_labels = cls_labels.to(device)
        masks = masks.to(device)

        cls_preds, decode_preds, cams = model(images, with_cam=True)
        cams = F.interpolate(
            cams, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        cams = F.relu(cams)
        cams = cams + F.adaptive_max_pool2d(-cams, (1, 1))
        cams /= F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5

        bkg = (
            torch.zeros(
                (cams.shape[0], 1, cams.shape[-2], cams.shape[-1]), device=device
            )
            + cfg.cam.bkg_score
        )
        pseudos = torch.argmax(torch.cat([bkg, cams], dim=1), dim=1)

        cls_loss = F.multilabel_soft_margin_loss(cls_preds, cls_labels)  ### cls loss
        decode_loss = F.cross_entropy(decode_preds, pseudos) + decode_criterion(
            decode_preds, pseudos
        )  ### seg loss

        agg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        overlap_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if cfg.train.use_aggregation_loss:
            agg_loss = aggregation_loss(decode_preds)  ### aggregation loss
        if cfg.train.use_overlapping_loss:
            overlap_loss = overlapping_loss(decode_preds)  ### overlap loss

        if epoch < cfg.train.no_seg_loss:
            loss = (
                1.0 * cls_loss + 0.0 * decode_loss + 0.0 * agg_loss + 0.0 * overlap_loss
            )
        else:
            if epoch < cfg.train.no_seg_loss * 2:
                loss = (
                    1.0 * cls_loss
                    + 0.1 * decode_loss
                    + 0.0 * agg_loss
                    + 0.0 * overlap_loss
                )
            else:
                loss = (
                    1.0 * cls_loss
                    + 0.1 * decode_loss
                    + 0.05 * agg_loss
                    + 0.05 * overlap_loss
                )

        with torch.no_grad():
            acc, recall = multi_label_accuracy(cls_preds, cls_labels)
            dice = dice_coef(decode_preds, masks, cfg.dataset.num_classes)
            avg_meter.add(
                {
                    "loss": loss.item(),
                    "cls_loss": cls_loss.item(),
                    "decode_loss": decode_loss.item(),
                    "agg_loss": agg_loss.item(),
                    "overlap_loss": overlap_loss.item(),
                    "acc": acc,
                    "recall": recall,
                    "dice": dice.item(),
                }
            )

        if i % cfg.train.log_step == 0:
            logging.info(
                "epoch:{:<5}/{:<5} ".format(epoch + 1, cfg.train.epochs)
                + "batch:{:<5}/{:<5} ".format(i + 1, total)
                + "lr:{:.6f} ".format(optimizers[0].param_groups[0]["lr"])
                + "loss:{:.6f} ".format(avg_meter.get("loss"))
                + "cls_loss:{:.6f} ".format(avg_meter.get("cls_loss"))
                + "decode_loss:{:.6f} ".format(avg_meter.get("decode_loss"))
                + "agg_loss:{:.6f} ".format(avg_meter.get("agg_loss"))
                + "overlap_loss:{:.6f} ".format(avg_meter.get("overlap_loss"))
                + "acc:{:.6f} ".format(avg_meter.get("acc"))
                + "recall:{:.6f} ".format(avg_meter.get("recall"))
                + "dice:{:.6f} ".format(avg_meter.get("dice"))
            )

        [optimizer.zero_grad() for optimizer in optimizers]
        loss.backward()
        [optimizer.step() for optimizer in optimizers]
    return (
        avg_meter.pop("loss"),
        avg_meter.pop("acc"),
        avg_meter.pop("recall"),
        avg_meter.pop("dice"),
    )


def validate(val_loader, model, epoch, decode_criterion):
    model.eval()  # switch to evaluate mode
    avg_meter = AverageMeter()
    with torch.no_grad():
        total = len(val_loader)
        for i, (images, cls_labels, masks, infos) in enumerate(val_loader):
            images = images.to(device)
            cls_labels = cls_labels.to(device)
            masks = masks.to(device)

            cls_preds, decode_preds, cams = model(images, with_cam=True)
            cams = F.interpolate(
                cams, size=images.shape[-2:], mode="bilinear", align_corners=False
            )
            cams = F.relu(cams)
            cams = cams + F.adaptive_max_pool2d(-cams, (1, 1))
            cams /= F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5
            bkg = (
                torch.zeros(
                    (cams.shape[0], 1, cams.shape[-2], cams.shape[-1]), device=device
                )
                + cfg.cam.bkg_score
            )
            pseudos = torch.argmax(torch.cat([bkg, cams], dim=1), dim=1)

            cls_loss = F.multilabel_soft_margin_loss(
                cls_preds, cls_labels
            )  ### cls loss
            decode_loss = F.cross_entropy(decode_preds, pseudos) + decode_criterion(
                decode_preds, pseudos
            )  ### seg loss

            agg_loss = torch.tensor(0.0, device=device, requires_grad=True)
            overlap_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if cfg.train.use_aggregation_loss:
                agg_loss = aggregation_loss(decode_preds)  ### aggregation loss
            if cfg.train.use_overlapping_loss:
                overlap_loss = overlapping_loss(decode_preds)  ### overlap loss

            if epoch < cfg.train.no_seg_loss:
                loss = (
                    1.0 * cls_loss
                    + 0.0 * decode_loss
                    + 0.0 * agg_loss
                    + 0.0 * overlap_loss
                )
            else:
                if epoch < cfg.train.no_seg_loss * 2:
                    loss = (
                        0.5 * cls_loss
                        + 0.5 * decode_loss
                        + 0.0 * agg_loss
                        + 0.0 * overlap_loss
                    )
                else:
                    if (
                        cfg.train.use_aggregation_loss
                        and cfg.train.use_overlapping_loss
                    ):
                        loss = (
                            0.4 * cls_loss
                            + 0.4 * decode_loss
                            + (0.1 * agg_loss + 0.1 * overlap_loss) / 2
                        )
                    else:
                        loss = (
                            0.4 * cls_loss
                            + 0.4 * decode_loss
                            + 0.1 * agg_loss
                            + 0.1 * overlap_loss
                        )

            acc, recall = multi_label_accuracy(cls_preds, cls_labels)
            dice = dice_coef(decode_preds, masks, cfg.dataset.num_classes)
            avg_meter.add(
                {
                    "loss": loss.item(),
                    "cls_loss": cls_loss.item(),
                    "decode_loss": decode_loss.item(),
                    "agg_loss": agg_loss.item(),
                    "overlap_loss": overlap_loss.item(),
                    "acc": acc,
                    "recall": recall,
                    "dice": dice.item(),
                }
            )

            if i % cfg.val.log_step == 0:
                logging.info(
                    "epoch:{:<5}/{:<5} ".format(epoch + 1, cfg.train.epochs)
                    + "batch:{:<5}/{:<5} ".format(i + 1, total)
                    + "loss:{:.6f} ".format(avg_meter.get("loss"))
                    + "val_cls_loss:{:.6f} ".format(avg_meter.get("cls_loss"))
                    + "val_decode_loss:{:.6f} ".format(avg_meter.get("decode_loss"))
                    + "val_agg_loss:{:.6f} ".format(avg_meter.get("agg_loss"))
                    + "val_overlap_loss:{:.6f} ".format(avg_meter.get("overlap_loss"))
                    + "val_acc:{:.6f} ".format(avg_meter.get("acc"))
                    + "val_recall:{:.6f} ".format(avg_meter.get("recall"))
                    + "val_dice:{:.6f} ".format(avg_meter.get("dice"))
                )
    return (
        avg_meter.pop("loss"),
        avg_meter.pop("acc"),
        avg_meter.pop("recall"),
        avg_meter.pop("dice"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="experiment name")
    parser.add_argument("--config", type=str, required=True, help="config file path")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="data file directory, example: ./data/brats2018/",
    )
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids, example: 0,1")
    parser.add_argument(
        "--arch",
        type=str,
        default="mit-b1",
        help="model architecture",
        choices=AVAILABLE_MODELS,
    )
    parser.add_argument(
        "--aggregation_loss",
        type=str,
        default="True",
        help="use aggregation loss",
        choices=("True", "False"),
    )
    parser.add_argument(
        "--overlapping_loss",
        type=str,
        default="True",
        help="use overlap loss",
        choices=("True", "False"),
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="initial weights path, use 'imagenet' if don't have self-supervised weights",
    )
    parser.add_argument("--resume", type=str, default=None, help="resuming weights dir")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="training batch size"
    )
    parser.add_argument("--epochs", type=int, default=None, help="training epochs")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    base_cfg = OmegaConf.load(cfg.parent)
    cfg = OmegaConf.merge(cfg, base_cfg)

    model_dir = "./middle/models/{}/".format(args.name)
    history_dir = "./middle/history/{}/".format(args.name)
    log_path = "./middle/logs/{}.log".format(args.name)

    cfg.train.use_overlapping_loss = True if args.overlapping_loss == "True" else False
    cfg.train.use_aggregation_loss = True if args.aggregation_loss == "True" else False
    cfg.dataset.txt_dir = args.data
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.epochs:
        cfg.train.epochs = args.epochs

    os.makedirs("./middle/logs/", exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )
    try:
        if torch.cuda.is_available() and args.gpus != "cpu":
            device = torch.device(f"cuda:{args.gpus}")
        else:
            device = torch.device("cpu")
        fix_seed()
        main()
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        exit(1)
