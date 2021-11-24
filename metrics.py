import numpy as np
import torch
from torch import Tensor, no_grad, nn

no_grad()
def calculate_acc(pred: Tensor, gt: Tensor) -> Tensor:
    return ((pred.sigmoid() > 0.5) == gt.bool()).float().mean()


def _calculate_binary_prep(pred: Tensor, gt: Tensor, threshold: float = -1.0, weights: Tensor = None) -> Tensor:
    if threshold < 0:
        pred = pred.sigmoid()
    else:
        pred = (pred.sigmoid() >= threshold).float()

    intersection_area = (pred * gt).sum(dim=(-1, -2))
    return pred, intersection_area


def _calculate_binary_iou(pred: Tensor, gt: Tensor, threshold: float = -1.0) -> Tensor:

    pred, intersection_area = _calculate_binary_prep(pred, gt, threshold)
    union_area = (pred + gt).clip(max=1).sum(dim=(-1, -2)).clip(min=1.)
    return intersection_area / union_area


def calculate_iou(pred: Tensor, gt: Tensor, threshold: float = -1.0) -> Tensor:
    return _calculate_binary_iou(pred, gt, threshold).mean()


def calculate_competion_iou(pred: Tensor, gt: Tensor) -> Tensor:

    iou = _calculate_binary_iou(pred, gt, threshold=0.5)

    score = 0
    for i, t in enumerate(np.arange(0.5, 1.0, 0.05)):
        score += ((iou > t).float().mean() - score) / (i + 1)

    return score


def calculate_binary_dice(pred: Tensor, gt: Tensor, threshold: float = -1, weights: Tensor = None) -> Tensor:
    pred, intersection_area = _calculate_binary_prep(pred, gt, threshold, weights)
    accumulated_area = (pred.sum(dim=(-1, -2)) + gt.sum(dim=(-1, -2))).clip(min=1.)
    return (2 * intersection_area / accumulated_area).mean()

def bce_weighted(pred, gt, weights):
    loss_func = nn.BCEWithLogitsLoss(reduction='none')
    loss_ = loss_func(pred, gt)
    return (loss_ * weights).sum(dim=(-1, -2)).mean()

@no_grad()
def inverse_frequency_weighting(base):
    mean_ = base.mean(dim=(-1, -2), keepdims=True)
    weights = 1 - ((mean_ * base) + ((1 - mean_) * (1 - base)))
    return weights / weights.sum(dim=(-1, -2), keepdims=True).clip(min=1.)
