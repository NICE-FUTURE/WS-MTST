import random
import math
import numpy as np
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as transforms
import torchvision.transforms.functional as transformsF


colors = list(matplotlib.colors.get_named_colors_mapping().values())


def fix_seed(seed=2):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def cal_eta(start_time, cur, total):
    time_now = datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total - cur) / float(cur)  # type: ignore
    delta = time_now - start_time
    eta = delta * scale
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(eta)


def plot_loss_history(losses, history_path):
    matplotlib.use("agg")
    plt.figure(figsize=(10, 8))
    plt.subplot(111)
    plt.plot(losses, color=colors[0], label="loss (self training)")
    plt.legend()
    plt.savefig(history_path)
    plt.close()


def plot_history(metrics, history_dir):
    matplotlib.use("agg")
    loss, acc, recall, dice, val_loss, val_acc, val_recall, val_dice = metrics

    plt.figure(figsize=(10, 8))
    plt.subplot(111)
    plt.plot(loss, color=colors[0], label="loss")
    plt.plot(val_loss, color=colors[0], label="val_loss", linestyle="--")
    plt.legend()
    plt.savefig("{}/loss.png".format(history_dir))
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(acc, color=colors[20], label="acc")
    plt.plot(val_acc, color=colors[20], label="val_acc", linestyle="--")
    plt.plot(recall, color=colors[50], label="recall")
    plt.plot(val_recall, color=colors[50], label="val_recall", linestyle="--")
    plt.legend()
    plt.subplot(212)
    plt.plot(dice, color=colors[40], label="dice")
    plt.plot(val_dice, color=colors[40], label="val_dice", linestyle="--")
    plt.legend()
    plt.savefig("{}/metrics.png".format(history_dir))
    plt.close()


def multi_label_accuracy(outputs, targets):
    """
    outputs: B, C
    targets: B, C
    """
    if len(outputs.shape) == 1:
        outputs = torch.unsqueeze(outputs, dim=0)
    outputs = outputs.sigmoid() > 0.5
    correct = ((outputs == targets) & (targets == 1)).sum().item()
    total = outputs.sum().item()
    true_total = (targets == 1).sum().item()
    if total > 0:
        accuracy = correct / total
    else:
        accuracy = 0
    if true_total > 0:
        recall = correct / true_total
    else:
        recall = 0
    return accuracy, recall


def dice_coef(output, target, n_classes):
    """calculate dice coefficient

    Args:
        output (torch.Tensor): BCHW
        target (torch.Tensor): BHW
        n_classes (int): include background `0`

    Returns:
        float: dice score
    """
    B, C, H, W = output.shape
    output = torch.softmax(output, dim=1).view(B, C, H * W)  # B C H W --> B C HW
    target_onehot = torch.zeros(
        target.shape[0],
        target.shape[1],
        target.shape[2],
        n_classes,
        device=output.device,
        dtype=output.dtype,
    )
    target_onehot = target_onehot.scatter(3, target.unsqueeze(-1), 1)
    target_onehot = target_onehot.permute((0, 3, 1, 2)).view(
        B, C, H * W
    )  # B H W --> B H W 1 --> B H W C --> B C H W --> B C HW

    score = torch.tensor(0, device=output.device, dtype=output.dtype)
    for i in range(C):
        intersection = (output[:, i, :] * target_onehot[:, i, :]).sum(dim=1)  # B
        summation = output[:, i, :].sum(dim=1) + target_onehot[:, i, :].sum(dim=1)  # B
        summation = torch.maximum(
            summation,
            torch.tensor(1.0 * 1e-5, device=output.device, dtype=output.dtype),
        )
        score += torch.mean((2.0 * intersection) / (summation))
    score /= C
    return score


def crf_inference_label(image, mask, n_labels, t=10, gt_prob=0.7):
    """
    `n_labels`: DO include `0` in counting labels.
    `t`: run `t` inference steps.

    return `result`: (H,W), processed by argmax.
    """
    from pydensecrf.utils import unary_from_labels
    import pydensecrf.densecrf as dcrf

    h, w = image.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)
    unary = unary_from_labels(mask, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)

    ### default sxy: 80, default srgb 13
    ### The names ('sxy' and 'srgb') are shorthand for "x/y standard-deviation" and "rgb standard-deviation"
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=image, compat=10)

    q = d.inference(t)

    return np.argmax(q, axis=0).reshape((h, w))


# from https://github.com/TACJu/TransFG/blob/master/utils/scheduler.py
class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step + 1) / float(max(1.0, self.warmup_steps + 1))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]  # type: ignore
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v


class ListMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = []

    def add(self, dict_):
        for k, v in dict_.items():
            if k not in self.__data:
                self.__data[k] = []
            self.__data[k].append(v)

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]]
        else:
            v_list = [self.__data[k] for k in keys]
            return tuple(v_list)

    def get_mean(self, *keys):
        if len(keys) == 1:
            return np.mean(self.__data[keys[0]])
        else:
            v_list = [np.mean(self.__data[k]) for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = []
        else:
            v = self.get(key)
            self.__data[key] = []
            return v


# transform functions
class ZeroOneNormalization(torch.nn.Module):
    """normalize a tensor to 0-1"""

    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): tensor to be normalized. (CHW format)

        Returns:
            Tensor: Normalized tensor.
        """
        C = tensor.shape[0]
        minimum = tensor.view((C, -1)).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        maximum = tensor.view((C, -1)).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        return (tensor - minimum + 1e-5) / (maximum - minimum + 1e-5)  # avoid Nan


class RandomHorizontalFlip(torch.nn.Module):
    """Modified for segmentation task
    Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, image, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return transformsF.hflip(image), transformsF.hflip(mask)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomVerticalFlip(torch.nn.Module):
    """Modified for segmentation task
    Vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, image, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return transformsF.vflip(image), transformsF.vflip(mask)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class Compose:
    """Modified for segmentation task
    Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class BraTSTransform(torch.nn.Module):
    """transform functions for training data of brats-like dataset."""

    def __init__(self, image_size, kernel_size=0.1, h_p=0.5, v_p=0.5):
        super().__init__()

        assert 0.0 < kernel_size < 1.0
        kernel_size = int(kernel_size * image_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.GaussianBlur(kernel_size=kernel_size),
                ZeroOneNormalization(),
            ]
        )
        self.both_transform = Compose(
            [RandomHorizontalFlip(p=h_p), RandomVerticalFlip(p=v_p)]
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        image = self.image_transform(image)
        mask = torch.from_numpy(mask)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(dim=0)
        image, mask = self.both_transform(image, mask)
        if len(mask.shape) == 3:
            mask = mask.squeeze()
        return image, mask
