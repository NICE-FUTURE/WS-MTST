import argparse
import os
import logging
import sys
import multiprocessing

import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from utils import crf_inference_label


def worker(i, total, name_id, image_dir, pred_dir, save_dir, palette):
    if i % 1000 == 0:
        logging.info("{}/{} {}".format(i, total, name_id))
    pred_path = os.path.join(pred_dir, "{}.png".format(name_id))
    image_path = os.path.join(image_dir, "{}.npy".format(name_id))
    pred = np.array(Image.open(pred_path), dtype=np.uint8)
    pred[pred == 4] = 3
    image = np.load(image_path)
    image = (image * 255).astype(np.uint8)
    image = (
        np.stack([image[:, :, 0], image[:, :, 2], image[:, :, 3]], axis=0)
        .transpose((1, 2, 0))
        .copy()
    )  # (H,W,3)
    result = crf_inference_label(image=image, mask=pred, n_labels=4, gt_prob=0.7)
    result[result == 3] = 4
    result = result.astype(np.uint8)
    save_path = os.path.join(save_dir, "{}.png".format(name_id))
    result = Image.fromarray(result, mode="L").convert("P")
    result.putpalette(palette)
    result.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file path")
    parser.add_argument(
        "--pred_dir", type=str, required=True, help="predicted mask directory"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    base_cfg = OmegaConf.load(cfg.parent)
    cfg = OmegaConf.merge(cfg, base_cfg)

    pred_dir = args.pred_dir
    pred_dir = pred_dir.replace("'", "").replace('"', "")

    base_dir = pred_dir[:-1] if pred_dir[-1] in ("/", "\\") else pred_dir
    save_dir = base_dir + "-crf"
    os.makedirs(save_dir, exist_ok=True)

    colormap_path = os.path.join(cfg.dataset.txt_dir, "grayscale_colormap.txt")
    with open(colormap_path, "r", encoding="utf-8") as f:
        palette = eval(f.read().strip())  # convert to list object

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    logging.info(str(sys.argv))

    # ================= multiprocessing post process =====================#
    image_dir = os.path.join(cfg.dataset.root_dir, "npy_images")
    name_ids = [os.path.splitext(item)[0] for item in os.listdir(pred_dir)]
    total = len(name_ids)
    num_cpus = min(multiprocessing.cpu_count(), 32)
    pool = multiprocessing.Pool(num_cpus)
    for idx, name_id in enumerate(name_ids):
        pool.apply_async(
            worker, args=(idx, total, name_id, image_dir, pred_dir, save_dir, palette)
        )
    pool.close()
    pool.join()
