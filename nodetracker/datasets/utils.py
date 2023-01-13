"""
Dataset utils
"""
from typing import List, Tuple

import torch
from torch.utils.data import default_collate


def ode_dataloader_collate_func(items: List[torch.Tensor]) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    ODE collate func: Standard way to batch sequences of dimension (T, *shape)
    where T is time dimension and shape is feature dimension is to create batch
    of size (B, T, *shape) but for NODE it makes more sense to do it as (T, B, *shape)
    which requires custom collate_func

    Args:
        items: Items gathered from WeatherDataset

    Returns:
        collated tensors
    """
    xs, x_ts, ys, y_ts, metadata = zip(*items)
    x, t_x, y, t_y = [torch.stack(v, dim=1) for v in [xs, x_ts, ys, y_ts]]
    metadata = default_collate(metadata)
    return x, t_x, y, t_y, metadata


def preprocess_batch(batch: tuple) -> tuple:
    """
    Unpacks batch and creates full trajectory tensors

    Args:
        batch: Raw batch

    Returns:
        Preprocessing batch
    """
    bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata = batch
    ts_all = torch.cat([ts_obs, ts_unobs], dim=0)
    bboxes_all = torch.cat([bboxes_obs, bboxes_unobs], dim=0)
    return bboxes_obs, bboxes_unobs, bboxes_all, ts_obs, ts_unobs, ts_all, metadata

