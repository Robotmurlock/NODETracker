"""
Implementation of GIoU loss hybrid/combinations with other loss functions.
"""
from typing import Dict

import torch
import torchvision
from torch import nn
from torch.nn import functional as F


def bbox_xyhw_to_xyxy(bbox: torch.Tensor, eps=1e-9) -> torch.Tensor:
    """
    Converts bbox format xyhw to xyxy.

    Args:
        bbox: BBox in xywh format.
        eps: minimum width and height

    Returns:
        Bbox in xyxy format
    """
    bbox[..., 2] = torch.max(bbox[..., 0] + bbox[..., 2], bbox[..., 0] + eps)
    bbox[..., 3] = torch.max(bbox[..., 1] + bbox[..., 3], bbox[..., 1] + eps)
    return bbox


def bounded_giou(
    bboxes1: torch.Tensor,
    bboxes2: torch.Tensor,
    reduction: str = 'mean',
    is_xyhw_format: bool = True,
    lower_bound: float = -1,
    upper_bound: float = 10,
    eps: float = 1e-9
) -> torch.Tensor:
    """
    Wrapper to torchvision GIoU loss. This loss can be unstable in case of weird bboxes dimensions (line).
    For that reasons lower and upper bound are introduced.
    Also, minimum weight and height are used (suppoerted by torchvision GIoU).

    Args:
        bboxes1: lhs bboxes
        bboxes2: rhs bboxes
        reduction: Loss batch reduction option (none, mean, sum)
        is_xyhw_format: If bbox is in xyhw format then it is converted to xyxy format
        lower_bound: Loss lower bound (clipped) - Makes loss more stable
        upper_bound: Loss upper bound (clipped) - Makes loss more stable
        eps: Minimum width/height - Makes loss more stable

    Returns:
        GIoU loss
    """
    if is_xyhw_format:
        # transform to xyxy format
        bboxes1 = bbox_xyhw_to_xyxy(bboxes1)
        bboxes2 = bbox_xyhw_to_xyxy(bboxes2)

    giou = torchvision.ops.generalized_box_iou_loss(bboxes1, bboxes2, reduction='none', eps=eps)
    giou = torch.clamp(giou, min=lower_bound, max=upper_bound)
    if reduction == 'none':
        return giou
    if reduction == 'mean':
        return giou.mean()
    if reduction == 'sum':
        return giou.sum()

    raise AssertionError(f'Unknown reduction option "{reduction}"!')


class HybridL1GIoU(nn.Module):
    """
    Combines L1 loss with GIoU loss.
    """
    def __init__(
        self,
        w_l1: float = 5,
        w_giou: float = 2,
        reduction: str = 'mean',
        is_xyhw_format: bool = True,
        giou_loss_lower_bound: float = -1,
        giou_loss_upper_bound: float = 10,
        eps: float = 1e-9,
        *args,
        **kwargs
    ):
        """
        Args:
            w_l1: L1 loss weight
            w_giou: GIoU loss weight
            reduction: Loss batch reduction option (none, mean, sum)
            is_xyhw_format: If bbox is in xyhw format then it is converted to xyxy format
            lower_bound: Loss lower bound (clipped) - Makes loss more stable
            upper_bound: Loss upper bound (clipped) - Makes loss more stable
            eps: Minimum width/height - Makes loss more stable
        """
        super().__init__(*args, **kwargs)
        self._w_l1 = w_l1
        self._w_giou = w_giou
        self._reduction = reduction
        self._is_xyhw_format = is_xyhw_format
        self._giou_loss_lower_bound = giou_loss_lower_bound
        self._giou_loss_upper_bound = giou_loss_upper_bound
        self._eps = eps

    def forward(self, bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1 = F.l1_loss(bboxes1, bboxes2, reduction=self._reduction)
        giou = bounded_giou(
            bboxes1=bboxes1,
            bboxes2=bboxes2,
            reduction=self._reduction,
            is_xyhw_format=self._is_xyhw_format,
            lower_bound=self._giou_loss_lower_bound,
            upper_bound=self._giou_loss_upper_bound,
            eps=self._eps
        )
        return {
            'loss': self._w_l1 * l1 + self._w_giou * giou,
            'l1': l1,
            'giou': giou
        }


class HybridGaussianNLLLossGIoU(nn.Module):
    """
    Combines GaussianNLLLoss loss with GIoU loss.
    """
    def __init__(
        self,
        w_nllloss: float = 4,
        w_giou: float = 1,
        reduction: str = 'mean',
        is_xyhw_format: bool = True,
        giou_loss_lower_bound: float = -1,
        giou_loss_upper_bound: float = 10,
        eps: float = 1e-9,
        *args,
        **kwargs
    ):
        """
        Args:
            w_nllloss: GaussianNLLLoss loss weight
            w_giou: GIoU loss weight
            reduction: Loss batch reduction option (none, mean, sum)
            is_xyhw_format: If bbox is in xyhw format then it is converted to xyxy format
            lower_bound: Loss lower bound (clipped) - Makes loss more stable
            upper_bound: Loss upper bound (clipped) - Makes loss more stable
            eps: Minimum width/height - Makes loss more stable
        """
        super().__init__(*args, **kwargs)
        self._w_nllloss = w_nllloss
        self._w_giou = w_giou
        self._reduction = reduction
        self._is_xyhw_format = is_xyhw_format
        self._giou_loss_lower_bound = giou_loss_lower_bound
        self._giou_loss_upper_bound = giou_loss_upper_bound
        self._eps = eps

    def forward(self, mean: torch.Tensor, gt: torch.Tensor, var: torch.Tensor) -> Dict[str, torch.Tensor]:
        nllloss = F.gaussian_nll_loss(mean, gt, var, reduction=self._reduction)
        giou = bounded_giou(
            bboxes1=mean,
            bboxes2=gt,
            reduction=self._reduction,
            is_xyhw_format=self._is_xyhw_format,
            lower_bound=self._giou_loss_lower_bound,
            upper_bound=self._giou_loss_upper_bound,
            eps=self._eps
        )
        return {
            'loss': self._w_nllloss * nllloss + self._w_giou * giou,
            'nllloss': nllloss,
            'giou': giou
        }


def run_test() -> None:
    pred = torch.tensor([
        # [0.1, 0.1, 0.2, 0.2]
        [0.0000, 0.0625, 0.0000, 0.0000]
    ], dtype=torch.float32)
    gt = torch.tensor([
        # [0.1, 0.1, 0.1, 0.1]
        [0.0000, 0.0638, 0.0000, 0.0000]
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
