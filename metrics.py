from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn, no_grad


@no_grad()
def calculate_acc(pred: Tensor, gt: Tensor) -> Tensor:
    return ((pred.sigmoid() > 0.5) == gt.bool()).float().mean()


def _calculate_binary_prep(
        pred: Tensor, gt: Tensor, threshold: float = -1.0, weights: Tensor = None
) -> Tuple[Tensor, Tensor]:
    """Helper method used for IoU and DICE calculations.

    Args:
        pred: the prediction. Expected as shape (B, *, W, H). Sigmoid is applied on tensor for squashing the values
        in between [0, 1].
        gt: the ground truth. Expected as same shape as pred.
        threshold: Threshold for discretizing the pred. If < 0, then the pred is not discretized.
        weights: Weights changing the contribution of each pixel (instead of uniform contribution). Not implemented
        yet.

    Returns:
        The pred in the [0, 1] range (discretized if threshold is positive).
        The intersection area between prediction and ground truth.
        The return shapes are [B, *].
    """

    if threshold < 0:
        pred = pred.sigmoid()
    else:
        pred = (pred.sigmoid() >= threshold).float()

    intersection_area = (pred * gt).sum(dim=(-1, -2))
    return pred, intersection_area


def _calculate_binary_iou(pred: Tensor, gt: Tensor, threshold: float = -1.0) -> Tensor:
    """Binary Intersection over Union for each item of the batch.

    Args:
        pred: the prediction. Expected as shape (B, *, W, H). Sigmoid is applied on tensor for squashing the values
        in between [0, 1].
        gt: the ground truth. Expected as same shape as pred.
        threshold: Threshold for discretizing the pred. If < 0, then the pred is not discretized.
    Returns:
        The IoU per item of the batch. Returns shape [B, *]
    """

    pred, intersection_area = _calculate_binary_prep(pred, gt, threshold)
    union_area = (pred + gt).clip(max=1).sum(dim=(-1, -2)).clip(min=1.0)
    return intersection_area / union_area


def calculate_iou(pred: Tensor, gt: Tensor, threshold: float = -1.0) -> Tensor:
    """Mean Binary Intersection over Union over the batch.

    Args:
        pred: the prediction. Expected as shape (B, *, W, H). Sigmoid is applied on tensor for squashing the values
        in between [0, 1].
        gt: the ground truth. Expected as same shape as pred.
        threshold: Threshold for discretizing the pred. If < 0, then the pred is not discretized.

    Returns:
        The average IoU of the batch. Returns shape [1].
    """
    return _calculate_binary_iou(pred, gt, threshold).mean()


def calculate_competion_iou(pred: Tensor, gt: Tensor) -> Tensor:
    """Score used in the competition for the leaderboard.

    :warning: **Not sure this method is implemented correctly. The description is quite critic.**
    More info about the score calculation: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/overview/evaluation

    Args:
        pred: the prediction. Expected as shape (B, *, W, H). Sigmoid is applied on tensor for squashing the values
        in between [0, 1].
        gt: the ground truth. Expected as same shape as pred.

    Returns:
        The IoU score of the batch. Returns shape [1].
    """

    iou = _calculate_binary_iou(pred, gt, threshold=0.5)

    score = 0
    for i, t in enumerate(np.arange(0.5, 1.0, 0.05)):
        score += ((iou > t).float().mean() - score) / (i + 1)

    return score


def calculate_binary_dice(
        pred: Tensor, gt: Tensor, threshold: float = -1, weights: Tensor = None
) -> Tensor:
    """Calculate dice score of the batch.

    Args:
        pred: the prediction. Expected as shape (B, *, W, H). Sigmoid is applied on tensor for squashing the values
        in between [0, 1].
        gt: the ground truth. Expected as same shape as pred.
        threshold: Threshold for discretizing the pred. If < 0, then the pred is not discretized.
        weights: Weights changing the contribution of each pixel (instead of uniform contribution). Not implemented
        yet.

    Returns:
        The DICE score of the batch. Returns shape [1].
    """

    pred, intersection_area = _calculate_binary_prep(pred, gt, threshold, weights)
    accumulated_area = (pred.sum(dim=(-1, -2)) + gt.sum(dim=(-1, -2))).clip(min=1.0)

    return (2 * intersection_area / accumulated_area).mean()


def bce_weighted(pred, gt, weights):
    """Pixel wise weighting of the Binary Cross Entropy Loss.

    Args:
         pred: the prediction. Expected as shape (B, *, W, H). Sigmoid is applied on tensor for squashing the values
        in between [0, 1].
        gt: the ground truth. Expected as same shape as pred.
        weights: Weights changing the contribution of each pixel (instead of uniform contribution). Not implemented
        yet.

    Returns:
        The weighted BCE loss of the batch. Returns shape [1].
    """

    loss_func = nn.BCEWithLogitsLoss(reduction="none")
    loss_ = loss_func(pred, gt)

    return (loss_ * weights).sum(dim=(-1, -2)).mean()


@no_grad()
def inverse_frequency_weighting(base: Tensor) -> Tensor:
    """Defines a pixel wise weighting.

    It aims to make both classes (background and foreground) as important from a loss perspective. The objective is
    to avoid biasing the model towards under-estimating foreground areas because they generally occupy less space in
    the images.

    This weighing is done per item in the batch.

    Args:
        base: Base for calculation of the weights. It should have the same shape as the batch (B, *, W, H).

    Returns:
        The pixel wise weighting in a tensor of same shape as the input (B, *, W, H).
    """
    mean_ = base.mean(dim=(-1, -2), keepdims=True)
    weights = 1 - ((mean_ * base) + ((1 - mean_) * (1 - base)))
    return weights / weights.sum(dim=(-1, -2), keepdims=True).clip(min=1.0)
