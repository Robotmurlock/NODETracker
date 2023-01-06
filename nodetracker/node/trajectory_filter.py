"""
Trajectory filter interface definition
"""
import torch
from typing import Optional, Tuple, Protocol


class TrajectoryFilter(Protocol):
    """
    Defines interface for bbox trajectory filter (forecasting)
    """
    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplemented('Forward method not implemented!')
