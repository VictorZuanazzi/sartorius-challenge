from typing import Optional

import torch
from torch import Tensor, no_grad
from torchvision.utils import save_image, make_grid


@no_grad()
def visualize_intersection(
    mask_pred: Tensor,
    mask_gt: Tensor,
    img: Optional[Tensor] = None,
    threshold: Optional[float] = None,
):

    if img is None:
        img = torch.zeros_like(mask_pred)

    if threshold is None:
        mask_pred = mask_pred.sigmoid()
    else:
        mask_pred = (mask_pred > threshold).float()

    intersection_raw = torch.cat((img, mask_pred, mask_gt), dim=1)
    tp = mask_pred * mask_gt
    fp = mask_pred * (1 - mask_gt)
    fn = (1 - mask_pred) * mask_gt
    mistakes = (fp + fn).clip(max=1)
    intersection_rgb = torch.cat((fp, tp, fn), dim=1)
    intersection_img = torch.cat((mistakes, tp, img))

    return intersection_raw, intersection_rgb, intersection_img
