import argparse
from datetime import datetime
import logging
import os
import time
import traceback
import copy

from omegaconf import OmegaConf

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from lightly.data import LightlyDataset
from lightly.loss import NegativeCosineSimilarity, NTXentLoss
from lightly.models.modules import (
    SimSiamProjectionHead,
    SimSiamPredictionHead,
    SimCLRProjectionHead,
)
from lightly.models.modules import (
    MoCoProjectionHead,
    BYOLProjectionHead,
    BYOLPredictionHead,
)
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.models.utils import random_token_mask

from datasets import SelfTrainBraTSDataset
from brats_collate import SimCLRCollateFunction, MoCoCollateFunction, MAECollateFunction
from network.model import Model
from network.lightly_utils import mask_at_index
from utils import (
    AverageMeter,
    ListMeter,
    WarmupCosineSchedule,
    cal_eta,
    fix_seed,
    plot_loss_history,
)


class SimCLR(nn.Module):
    def __init__(self, backbone, input_dim, pooling="gmp"):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(
            input_dim=input_dim, hidden_dim=512, output_dim=128
        )
        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d
        self.criterion = NTXentLoss()

    def forward(self, x):
        _x = self.backbone(x)  # [_x1, _x2, _x3, _x4], single shape: B,D,H,W
        x = self.pooling(_x[-1], (1, 1)).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch):
        (x0, x1), _, _ = batch
        x0 = x0.to(device)
        x1 = x1.to(device)

        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self, lr=0.06):
        optim = torch.optim.SGD(self.parameters(), lr)
        return optim

    def configure_collect_fn(self, img_size):
        collect_fn = SimCLRCollateFunction(
            input_size=img_size, vf_prob=0.5, hf_prob=0.5
        )
        return collect_fn


class SimSiam(nn.Module):
    def __init__(
        self, backbone, input_dim, output_dim=128, hidden_dim=64, pooling="gmp"
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(input_dim, input_dim, output_dim)
        self.prediction_head = SimSiamPredictionHead(output_dim, hidden_dim, output_dim)
        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        _x = self.backbone(x)  # [_x1, _x2, _x3, _x4], single shape: B,D,H,W
        x = self.pooling(_x[-1], (1, 1)).flatten(start_dim=1)
        z = self.projection_head(x)
        p = self.prediction_head(z)
        if torch.isnan(x.mean()):
            print(
                "x:{}, f:{}; z:{}, p:{}.".format(
                    x.mean().item(), x.mean().item(), z.mean().item(), p.mean().item()
                )
            )
            raise Exception("[in SimSiam.forward()]: Nan loss detected.")
        z = z.detach()
        return z, p

    def training_step(self, batch):
        (x0, x1), _, _ = batch
        x0 = x0.to(device)
        x1 = x1.to(device)

        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self, lr=0.06):
        optim = torch.optim.SGD(self.parameters(), lr=lr)
        return optim

    def configure_collect_fn(self, img_size):
        collect_fn = SimCLRCollateFunction(
            input_size=img_size, vf_prob=0.5, hf_prob=0.5
        )
        return collect_fn


class MoCo(nn.Module):
    def __init__(self, backbone, input_dim, pooling="gmp"):
        super().__init__()
        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(
            input_dim=input_dim, hidden_dim=512, output_dim=128
        )

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.criterion = NTXentLoss(memory_bank_size=4096)

    def forward(self, x):
        _x = self.backbone(x)  # [_x1, _x2, _x3, _x4], single shape: B,D,H,W
        query = self.pooling(_x[-1], (1, 1)).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        _x = self.backbone_momentum(x)  # [_x1, _x2, _x3, _x4], single shape: B,D,H,W
        key = self.pooling(_x[-1], (1, 1)).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def training_step(self, batch):
        update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, m=0.99)
        (x_query, x_key), _, _ = batch
        x_query = x_query.to(device)
        x_key = x_key.to(device)

        query = self.forward(x_query)
        key = self.forward_momentum(x_key)
        loss = self.criterion(query, key)
        return loss

    def configure_optimizers(self, lr=0.06):
        optim = torch.optim.SGD(self.parameters(), lr=lr)
        return optim

    def configure_collect_fn(self, img_size):
        collect_fn = MoCoCollateFunction(input_size=img_size, vf_prob=0.5, hf_prob=0.5)
        return collect_fn


class BYOL(nn.Module):
    def __init__(
        self, backbone, input_dim, output_dim=128, hidden_dim=512, pooling="gmp"
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(input_dim, hidden_dim, output_dim)
        self.prediction_head = BYOLPredictionHead(output_dim, hidden_dim, output_dim)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        _x = self.backbone(x)  # [_x1, _x2, _x3, _x4], single shape: B,D,H,W
        y = self.pooling(_x[-1], (1, 1)).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        _x = self.backbone(x)  # [_x1, _x2, _x3, _x4], single shape: B,D,H,W
        y = self.pooling(_x[-1], (1, 1)).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch):
        update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, m=0.99)
        (x0, x1), _, _ = batch
        x0 = x0.to(device)
        x1 = x1.to(device)

        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        return loss

    def configure_optimizers(self, lr=0.06):
        return torch.optim.SGD(self.parameters(), lr=lr)

    def configure_collect_fn(self, img_size):
        collect_fn = SimCLRCollateFunction(
            input_size=img_size, vf_prob=0.5, hf_prob=0.5
        )
        return collect_fn


class SimMIM(nn.Module):
    def __init__(
        self,
        vit,
        first_dim,
        last_dim,
        first_num_patch,
        last_num_patch,
        in_channel,
        image_size,
        mask_ratio=0.75,
    ):
        super().__init__()

        self.seq_length = first_num_patch
        self.hidden_dim = last_dim
        self.mask_ratio = mask_ratio
        self.in_channel = in_channel
        self.image_size = image_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, first_dim))  # type: ignore

        self.backbone_embedding = vit.patch_embed1
        self.backbone = vit

        self.decoder = nn.Linear(
            self.hidden_dim, image_size**2 * in_channel // last_num_patch
        )
        self.pixel_shuffle = nn.PixelShuffle(image_size // int(last_num_patch**0.5))
        self.criterion = nn.L1Loss()

    def forward_encoder(self, x, idx_mask):
        # pass all the tokens to the encoder, both masked and non masked ones
        x, H, W = self.backbone_embedding(x)
        x = mask_at_index(x, idx_mask, self.mask_token)
        x = self.backbone((x, H, W), is_embedding=True)
        B, C, H, W = x[-1].shape
        out = x[-1].view(B, C, -1).permute((0, 2, 1))
        return out

    def forward_decoder(self, x_encoded):
        return self.decoder(x_encoded)

    def forward(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = random_token_mask(
            size=(batch_size, self.seq_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )

        # Encoding...
        x_encoded = self.forward_encoder(
            images, idx_mask
        )  # B L1 C1 | B C H W --> B L2 C2

        # Decoding...
        x_out = self.forward_decoder(x_encoded)  # B L2 C2 --> B L2 C3 | B L2 HWC//L2
        B, L, C = x_out.shape

        # B L2 C3 | B L2 HWC//L2 --> B C H W
        x_out = x_out.permute((0, 2, 1)).view(B, C, int(L**0.5), int(L**0.5))
        x_out = self.pixel_shuffle(x_out)

        return x_out, images

    def training_step(self, batch):
        x, _, _ = batch
        x = x.to(device)
        predictions, targets = self.forward(x)
        loss = self.criterion(predictions, targets)
        return loss

    def configure_optimizers(self, lr=0.06):
        return torch.optim.AdamW(self.parameters(), lr=lr)

    def configure_collect_fn(self, img_size):
        collect_fn = MAECollateFunction(input_size=img_size)  # type: ignore
        return collect_fn


def main():
    MODELS = {
        "SimCLR": SimCLR,
        "SimSiam": SimSiam,
        "MoCo": MoCo,
        "BYOL": BYOL,
        "SimMIM": SimMIM,
    }

    ### model
    backbone = Model(
        backbone=args.arch,
        num_classes=cfg.dataset.num_classes,
        decoder_dim=256,
        pretrained=False,
        num_channels=cfg.dataset.num_channels,
        out_size=cfg.dataset.img_size,
    ).encoder

    assert args.method in MODELS, "method '{}' is not implemented.".format(args.method)
    if args.method == "SimMIM":
        model = SimMIM(
            backbone,
            first_dim=backbone.embed_dims[0],
            last_dim=backbone.embed_dims[-1],
            first_num_patch=(cfg.dataset.img_size // 4) ** 2,
            last_num_patch=(cfg.dataset.img_size // 16) ** 2,
            in_channel=cfg.dataset.num_channels,
            image_size=cfg.dataset.img_size,
        )
    else:
        model = MODELS[args.method](backbone, input_dim=backbone.embed_dims[-1])
    model.to(device)
    logging.info("\nargs: {}".format(args))
    logging.info("\nconfigs: {}".format(cfg))

    ### dataset & dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = SelfTrainBraTSDataset(
        txt_dir=cfg.dataset.txt_dir,
        root_dir=cfg.dataset.root_dir,
        stage=cfg.pretrain.stage,
    )

    def index_to_filename(inner_dataset, index):
        return inner_dataset.name_list[index]

    dataset = LightlyDataset.from_torch_dataset(
        dataset, transform=transform, index_to_filename=index_to_filename
    )
    collate_fn = model.configure_collect_fn(img_size=cfg.dataset.img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.pretrain.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    ### loss function & optimizer & learning rate scheduler
    optimizer = model.configure_optimizers(lr=cfg.pretrain.learning_rate)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=cfg.pretrain.warmup_epochs,
        t_total=int(1.1 * cfg.pretrain.epochs),
    )

    logging.info("START TIME:{}".format(time.asctime(time.localtime(time.time()))))
    start_time = datetime.now().replace(microsecond=0)
    best = None
    list_meter = ListMeter()
    for epoch in range(cfg.pretrain.epochs):
        # train
        model.train()
        batch_avg_meter = AverageMeter()
        total = len(dataloader)
        for i, batch in enumerate(dataloader):
            batch_loss = model.training_step(batch)
            batch_avg_meter.add({"loss": batch_loss.item()})
            if i % cfg.pretrain.log_step == 0:
                logging.info(
                    "epoch:{:<5}/{:<5} ".format(epoch + 1, cfg.pretrain.epochs)
                    + "batch:{:<5}/{:<5} ".format(i + 1, total)
                    + "lr:{:.6f} ".format(optimizer.param_groups[0]["lr"])
                    + "loss:{:.6f} ".format(batch_loss)
                )
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        scheduler.step()
        loss = batch_avg_meter.pop("loss")
        list_meter.add({"loss": loss})

        eta = cal_eta(start_time, epoch + 1, cfg.pretrain.epochs)
        logging.info(
            "epoch:{:<5}/{:<5} ".format(epoch + 1, cfg.pretrain.epochs)
            + "eta:{} ".format(eta)
            + "lr:{:.6f} ".format(optimizer.param_groups[0]["lr"])
            + "loss:{:.6f}".format(loss)
        )
        plot_loss_history(list_meter.get("loss"), history_path)

        # save best model
        torch.save(model.state_dict(), model_dir + "best.pth")
        if best is None or loss < best:
            best = loss
            torch.save(model.state_dict(), model_dir + "last.pth")
            logging.info("Saved best model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="experiment name")
    parser.add_argument("--config", type=str, required=True, help="config file path")
    parser.add_argument(
        "--method",
        type=str,
        default="SimSiam",
        choices=("SimCLR", "SimSiam", "MoCo", "BYOL", "SimMIM"),
        help="implementation of contrastive learning",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="mit-b1",
        help="model architecture",
        choices=["mit-b0", "mit-b1", "mit-b2", "mit-b3", "mit-b4", "mit-b5"],
    )
    parser.add_argument("--epochs", type=int, default=None, help="epoches")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size")
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids, example: 0,1")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    base_cfg = OmegaConf.load(cfg.parent)
    cfg = OmegaConf.merge(cfg, base_cfg)
    if args.epochs:
        cfg.pretrain.epochs = args.epochs
    if args.batch_size:
        cfg.pretrain.batch_size = args.batch_size

    model_dir = "./middle/pretrain/models/{}/".format(args.name)
    history_path = "./middle/pretrain/history/{}.png".format(args.name)
    log_path = "./middle/pretrain/logs/{}.log".format(args.name)

    os.makedirs("./middle/pretrain/logs/", exist_ok=True)
    os.makedirs("./middle/pretrain/history/", exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
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
