"""
Dataset utils
"""
from typing import List, Tuple
import torch


def ode_dataloader_collate_func(items: List[torch.Tensor]) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    xs, x_ts, ys, y_ts = zip(*items)
    x, t_x, y, t_y = torch.stack(xs, dim=1), torch.stack(x_ts, dim=1), torch.stack(ys, dim=1), torch.stack(y_ts, dim=1)
    return x, t_x, y, t_y
