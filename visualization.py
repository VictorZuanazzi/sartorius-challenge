from typing import Optional

import torch
from torch import Tensor, no_grad
from torchvision.utils import make_grid, save_image


@no_grad()
def visualize_intersection(
    mask_pred: Tensor,
    mask_gt: Tensor,
    img: Optional[Tensor] = None,
    threshold: Optional[float] = None,
):
    """Create useful visualizations based on prediction and ground truth.

    All tensor inputs should have same shape (B, 1, W, H).
    Args:
        mask_pred: predicted mask.
        mask_gt: ground truth mask.
        img: image.
        threshold: threshold for discretizing mask_pred.

    Returns:
        3 images.
    """

    if img is None:
        img = torch.zeros_like(mask_pred)

    if threshold is None:
        mask_pred = mask_pred.sigmoid()
    else:
        mask_pred = (mask_pred > threshold).float()

    # Just shows the intersection of pred and gt using the channels.
    intersection_raw = torch.cat((img, mask_pred, mask_gt), dim=1)

    # use the channels for showing true-positves (tp: green), false-positives (fp: red) and false-negatives (fn: blue)
    tp = mask_pred * mask_gt
    fp = mask_pred * (1 - mask_gt)
    fn = (1 - mask_pred) * mask_gt
    intersection_rgb = torch.cat((fp, tp, fn), dim=1)

    # compresses mistakes into the red channel and true positives on the green channel. The blue channel is used to
    # overlap the original image.
    mistakes = (fp + fn).clip(max=1)
    intersection_img = torch.cat((mistakes, tp, img))

    return intersection_raw, intersection_rgb, intersection_img
