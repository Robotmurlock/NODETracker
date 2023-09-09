from typing import Tuple

import torch


def to_scaled_relative_ts(ts_obs: torch.Tensor, ts_unobs: torch.Tensor, t_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # Relative time points
    ts_last = ts_obs[-1, 0, 0]
    ts_obs = ts_obs - ts_last
    ts_unobs = ts_unobs - ts_last

    # Scaling
    ts_obs = ts_obs / t_scale
    ts_unobs = ts_unobs / t_scale

    return ts_obs, ts_unobs