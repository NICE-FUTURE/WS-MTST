from pathlib import Path
import time
import logging
import argparse
import os
from itertools import chain

import numpy as np
import cv2
from PIL import Image
from omegaconf import OmegaConf
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F

from datasets import BraTSDataset
from utils import fix_seed
from network.model import Model, AVAILABLE_MODELS


# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def generate_relevance(model, input, num_tokens, index=None):
    output = model(input, register_hook=True)  # [1, 3]  # batch_size:1, num_classes:3
    if index == None:
        index = np.argmax(output.detach().cpu().numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.to(device) * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    R = torch.eye(num_tokens, num_tokens).to(device)
    blocks = chain(
        model.encoder.block1,
        model.encoder.block2,
        model.encoder.block3,
        model.encoder.block4,
    )
    for blk in blocks:
        grad = blk.attn.get_attn_gradients()
        grad = F.interpolate(grad, (num_tokens, num_tokens), mode="bilinear")
        cam = blk.attn.get_attention_map()
        cam = F.interpolate(cam, (num_tokens, num_tokens), mode="bilinear")
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R, cam)
    R = R.mean(dim=1)
    return R


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def visualize_attention_map(model, image, height, width, class_idx=None):
    sqrt_num_token = 10
    attn_map = generate_relevance(
        model,
        input=image.unsqueeze(dim=0),
        num_tokens=sqrt_num_token**2,
        index=class_idx,
    )
    attn_map = attn_map.reshape(1, 1, sqrt_num_token, sqrt_num_token)
    attn_map = torch.nn.functional.interpolate(
        attn_map, (height, width), mode="bilinear"
    )
    attn_map = attn_map.reshape(height, width).detach().cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    image_attn_map = np.concatenate(
        [image.permute(1, 2, 0).detach().cpu().numpy()[..., 0:1]] * 3, axis=2
    )
    image_attn_map = (image_attn_map - image_attn_map.min()) / (
        image_attn_map.max() - image_attn_map.min()
    )
    vis = show_cam_on_image(image_attn_map, attn_map)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


# @torch.no_grad()  # comment when visualizing attention maps
def main():
    ### model
    model = Model(
        backbone=args.arch,
        num_classes=cfg.dataset.num_classes,
        decoder_dim=256,
        pretrained=False,
        num_channels=cfg.dataset.num_channels,
        out_size=cfg.dataset.img_size,
    )
    state_dict = dict(torch.load(args.model_path, map_location=device))
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not (key.endswith("total_params") or key.endswith("total_ops"))
    }
    model.load_state_dict(state_dict)
    logging.info("\nargs: {}".format(args))
    logging.info("\nconfigs: {}".format(cfg))
    model.to(device)

    ### dataset (batch_size: 1)
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    test_dataset = BraTSDataset(
        txt_dir=cfg.dataset.txt_dir,
        root_dir=cfg.dataset.root_dir,
        stage=cfg.test.stage,
        transform=test_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.test.num_workers,
        pin_memory=True,
    )

    ### test
    logging.info("START TIME:{}".format(time.asctime(time.localtime(time.time()))))
    model.eval()  # switch to evaluate mode
    total = len(test_loader)
    for j, (images, cls_labels, masks, infos) in enumerate(test_loader):
        images = images.to(device)
        img_name = [item + ".png" for item in infos["img_name"]]
        height, width = images.shape[-2:]

        ### save attention visualizations
        for image, name in zip(images, img_name):
            vis0 = visualize_attention_map(model, image, height, width, class_idx=0)
            vis1 = visualize_attention_map(model, image, height, width, class_idx=1)
            vis2 = visualize_attention_map(model, image, height, width, class_idx=2)
            vis_list = (vis0, vis1, vis2, np.zeros_like(vis0))

            temp_image = image.permute(1, 2, 0).detach().cpu().numpy()
            fig, axs = plt.subplots(
                nrows=2,
                ncols=4,
                figsize=(8, 4),
                gridspec_kw={"wspace": 0, "hspace": 0.1},
            )
            for i in range(4):
                axs[0, i].imshow(temp_image[..., i : i + 1], cmap="gray")
                axs[0, i].axis("off")
            for i in range(4):
                axs[1, i].imshow(vis_list[i])
                axs[1, i].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(attns_save_dir, name), bbox_inches="tight")
            plt.close()

        cls_preds, decode_preds, cams = model(images, with_cam=True)

        ### save pred mask
        pred = torch.argmax(decode_preds, dim=1).cpu().numpy().astype(np.uint8)
        pred[pred == 3] = 4
        for item, name in zip(pred, img_name):
            item = Image.fromarray(item, mode="L").convert("P")
            item.putpalette(palette)
            item.save(os.path.join(preds_save_dir, name))

        ### save pseudo mask
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
        pseudo = (
            torch.argmax(torch.cat([bkg, cams], dim=1), dim=1)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        pseudo[pseudo == 3] = 4
        for item, name in zip(pseudo, img_name):
            item = Image.fromarray(item, mode="L").convert("P")
            item.putpalette(palette)
            item.save(os.path.join(pseudos_save_dir, name))

        if j % cfg.test.log_step == 0:
            logging.info("batch:{:<5}/{:<5}".format(j + 1, total))

    logging.info("STOP TIME:{}".format(time.asctime(time.localtime(time.time()))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file path")
    parser.add_argument(
        "--model_path", type=str, required=True, help=".pth model file path"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="mit-b1",
        help="model architecture",
        choices=AVAILABLE_MODELS,
    )
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids, example: 0,1")
    args = parser.parse_args()
    # args.name = os.path.splitext(os.path.basename(args.model_path))[0]
    args.name = Path(args.model_path).parent.name
    cfg = OmegaConf.load(args.config)
    base_cfg = OmegaConf.load(cfg.parent)
    cfg = OmegaConf.merge(cfg, base_cfg)

    pseudos_save_dir = "./middle/test/{}/pseudos/".format(args.name)
    preds_save_dir = "./middle/test/{}/preds/".format(args.name)
    attns_save_dir = "./middle/test/{}/attns/".format(args.name)

    colormap_path = os.path.join(cfg.dataset.txt_dir, "grayscale_colormap.txt")
    with open(colormap_path, "r", encoding="utf-8") as f:
        palette = eval(f.read().strip())  # convert to list object

    for _ in (pseudos_save_dir, preds_save_dir, attns_save_dir):
        os.makedirs(_, exist_ok=True)
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    if torch.cuda.is_available() and args.gpus != "cpu":
        device = torch.device(f"cuda:{args.gpus}")
    else:
        device = torch.device("cpu")
    fix_seed()
    main()
