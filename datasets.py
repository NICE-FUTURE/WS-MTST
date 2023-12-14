import os

import numpy as np

import torch
from torch.utils.data import Dataset


def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list


def load_cls_label_list(txt_dir):
    return np.load(
        os.path.join(txt_dir, "cls_labels_onehot.npy"), allow_pickle=True
    ).item()


class BraTSDataset(Dataset):
    def __init__(self, txt_dir="", root_dir="", stage="", transform=None):
        super().__init__()

        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "npy_images")
        self.mask_dir = os.path.join(root_dir, "npy_masks")
        self.transform = transform
        self.stage = stage

        self.txt_path = os.path.join(txt_dir, stage + ".txt")
        self.name_list = load_img_name_list(self.txt_path)
        self.label_list = load_cls_label_list(
            txt_dir=txt_dir
        )  # dict: {name: one-hot label}

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx]
        img_path = os.path.join(self.img_dir, img_name + ".npy")
        image = np.load(img_path, allow_pickle=True)
        origin_size = image.shape

        mask_path = os.path.join(self.mask_dir, img_name + ".npy")
        mask = np.load(mask_path, allow_pickle=True).astype(np.int64)
        mask[mask == 4] = 3

        if self.transform:
            if self.stage == "train":
                image, mask = self.transform(image, mask)
            else:
                image = self.transform(image)
        image = image.to(torch.float32)

        cls_label = self.label_list[img_name]

        return (
            image,
            cls_label,
            mask,
            {"img_name": img_name, "origin_size": origin_size},
        )


class SelfTrainBraTSDataset(Dataset):
    def __init__(self, txt_dir="", root_dir="", stage="", transform=None):
        super().__init__()

        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "npy_images")
        self.transform = transform

        self.txt_path = os.path.join(txt_dir, stage + ".txt")
        self.name_list = load_img_name_list(self.txt_path)
        self.label_list = load_cls_label_list(
            txt_dir=txt_dir
        )  # dict: {name: one-hot label}

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx]
        img_path = os.path.join(self.img_dir, img_name + ".npy")
        image = np.load(img_path, allow_pickle=True)
        if self.transform:
            # np.ndarray to torch.Tensor
            image = self.transform(image).to(torch.float32)
        cls_label = self.label_list[img_name]

        return image, cls_label
