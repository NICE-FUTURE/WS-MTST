from typing import Any

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def aggregation_loss(preds: torch.Tensor):
    """
    Args:
        preds (torch.Tensor): (B, C+1, H, W) include background

    Returns:
        float: aggregation loss
    """
    B, D, H, W = preds.shape
    preds = torch.sigmoid(preds[:, 1:]).view(B, D - 1, H * W)  # (B,C,H*W)
    preds = torch.where(preds < 0.5, torch.zeros_like(preds), preds)

    hv, wv = torch.meshgrid(
        [torch.arange(H, device=preds.device), torch.arange(W, device=preds.device)]
    )  # (H,W) (H,W) dtype: Long
    hv, wv = hv.unsqueeze(0).repeat(B, 1, 1), wv.unsqueeze(0).repeat(
        B, 1, 1
    )  # (B,H,W) (B,H,W)
    dist = (torch.square(hv) + torch.square(wv)).view(B, H * W)  # (B,H*W)
    dist_min, dist_max = (
        torch.min(dist, dim=1, keepdim=True)[0],
        torch.max(dist, dim=1, keepdim=True)[0],
    )
    dist = ((dist - dist_min) / (dist_max - dist_min)).to(preds.dtype)

    dist0 = preds[:, 0] * dist
    dist1 = preds[:, 1] * dist
    dist2 = preds[:, 2] * dist
    loss0 = torch.mean(
        torch.sqrt(
            torch.mean(
                torch.square(dist0 - torch.mean(dist0, dim=1, keepdim=True)), dim=1
            )
        )
    )
    loss1 = torch.mean(
        torch.sqrt(
            torch.mean(
                torch.square(dist1 - torch.mean(dist1, dim=1, keepdim=True)), dim=1
            )
        )
    )
    loss2 = torch.mean(
        torch.sqrt(
            torch.mean(
                torch.square(dist2 - torch.mean(dist2, dim=1, keepdim=True)), dim=1
            )
        )
    )
    loss = (loss0 + loss1 + loss2) / 3
    return loss


def overlapping_loss(preds: torch.Tensor):
    """
    Args:
        preds (torch.Tensor): (B, C+1, H, W) include background

    Returns:
        float: overlapping loss
    """
    B, D, H, W = preds.shape
    preds = torch.sigmoid(preds[:, 1:]).view(B, D - 1, H * W)  # (B,C,H*W)
    preds = torch.where(preds < 0.5, torch.zeros_like(preds), preds)

    loss1_3 = 1 - torch.mean(
        (torch.sum(preds[:, 0] * preds[:, 2], dim=1))
        / (torch.sum(preds[:, 2], dim=1) + 1e-5),
        dim=0,
    )
    loss1_2 = 1 - torch.mean(
        (torch.sum(preds[:, 0] * preds[:, 1], dim=1))
        / (torch.sum(preds[:, 1], dim=1) + 1e-5),
        dim=0,
    )
    loss3_2 = 1 - torch.mean(
        (torch.sum(preds[:, 2] * preds[:, 1], dim=1))
        / (torch.sum(preds[:, 1], dim=1) + 1e-5),
        dim=0,
    )
    loss = (loss1_3 + loss1_2 + loss3_2) / 3

    return loss


class FocalLoss(nn.Module):
    """
    borrowed from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss/focal_loss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor
        gamma:
        ignore_index:
        reduction:
    """

    def __init__(
        self, num_class, weights=None, gamma=2, ignore_index=None, reduction="mean"
    ):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.ignore_index = ignore_index
        self.alpha: Any = weights
        if self.alpha is None:
            self.alpha = torch.ones(
                num_class,
            )
        elif isinstance(self.alpha, (int, float)):
            self.alpha = torch.as_tensor([self.alpha] * num_class)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(self.alpha)
        else:
            raise NotImplementedError
        if self.alpha.shape[0] != num_class:
            raise RuntimeError("the length not equal to number of class")

    def forward(self, logit, target):
        N, C = logit.shape[:2]
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)
        if prob.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        ori_shp = target.shape
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]
        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target * valid_mask

        # ----------memory saving way--------
        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        alpha_class = alpha[target.squeeze().long()]
        class_weight = -alpha_class * torch.pow(torch.sub(1.0, prob), self.gamma)
        loss = class_weight * logpt
        if valid_mask is not None:
            loss = loss * valid_mask.squeeze()

        if self.reduction == "mean":
            loss = loss.mean()
            if valid_mask is not None:
                loss = loss.sum() / valid_mask.sum()
        elif self.reduction == "none":
            loss = loss.view(ori_shp)
        return loss
