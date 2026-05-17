"""多类（含二分类）Focal Loss，与 CrossEntropy 接口一致：forward(logits, target)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_ce_or_focal(loss_name: str, gamma: float = 2.0, *, reduction: str = "mean") -> nn.Module:
    """训练脚本共用：ce | focal。reduction 支持 mean | none（none 时配合 per-sample 权重）。"""
    ln = (loss_name or "ce").strip().lower()
    if ln == "focal":
        return FocalLoss(gamma=float(gamma), reduction=reduction)
    return nn.CrossEntropyLoss(reduction=reduction)
