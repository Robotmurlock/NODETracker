"""
Implementation of GIoU loss hybrid/combinations with other loss functions.
"""
from typing import Dict

import torch
import torchvision
from torch import nn
from torch.nn import functional as F


def bbox_xyhw_to_xyxy(bbox: torch.Tensor) -> torch.Tensor:
    """
    Converts bbox format xyhw to xyxy.

    Args:
        bbox: BBox in xywh format.

    Returns:
        Bbox in xyxy format
    """
    bbox[..., 2] = bbox[..., 0] + bbox[..., 2]
    bbox[..., 3] = bbox[..., 1] + bbox[..., 3]
    bbox = torch.clip(bbox, min=0.0, max=1.0)
    return bbox


class HybridL1GIoU(nn.Module):
    def __init__(
        self,
        w_l1: float = 5,
        w_giou: float = 2,
        reduction: str = 'mean',
        is_xyhw_format: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._w_l1 = w_l1
        self._w_giou = w_giou
        self._reduction = reduction
        self._is_xyhw_format = is_xyhw_format

    def forward(self, bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1 = F.l1_loss(bboxes1, bboxes2, reduction=self._reduction)
        if self._is_xyhw_format:
            # transform to xyxy format
            bboxes1 = bbox_xyhw_to_xyxy(bboxes1)
            bboxes2 = bbox_xyhw_to_xyxy(bboxes2)

        giou = torchvision.ops.generalized_box_iou_loss(bboxes1, bboxes2, reduction=self._reduction)
        return {
            'loss': self._w_l1 * l1 + self._w_giou * giou,
            'l1': l1,
            'giou': giou
        }


class HybridGaussianNLLLossGIoU(nn.Module):
    def __init__(
        self,
        w_nllloss: float = 4,
        w_giou: float = 1,
        reduction: str = 'mean',
        is_xyhw_format: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._w_nllloss = w_nllloss
        self._w_giou = w_giou
        self._reduction = reduction
        self._is_xyhw_format = is_xyhw_format

    def forward(self, mean: torch.Tensor, gt: torch.Tensor, var: torch.Tensor) -> Dict[str, torch.Tensor]:
        nllloss = F.gaussian_nll_loss(mean, gt, var, reduction=self._reduction)
        if self._is_xyhw_format:
            # transform to xyxy format
            mean = bbox_xyhw_to_xyxy(mean)
            gt = bbox_xyhw_to_xyxy(gt)

        giou = torchvision.ops.generalized_box_iou_loss(mean, gt, reduction=self._reduction)
        return {
            'loss': self._w_nllloss * nllloss + self._w_giou * giou,
            'nllloss': nllloss,
            'giou': giou
        }


def run_test() -> None:
    pred = torch.tensor([
        # [0.1, 0.1, 0.2, 0.2]
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32)
    gt = torch.tensor([
        # [0.1, 0.1, 0.1, 0.1]
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32)
    var = torch.tensor([
        [1.0, 1.0, 1.0, 1.0]
    ], dtype=torch.float32)

    loss = HybridL1GIoU()
    print(loss(pred, gt))
    loss = HybridGaussianNLLLossGIoU()
    print(loss(pred, gt, var))


if __name__ == '__main__':
    run_test()