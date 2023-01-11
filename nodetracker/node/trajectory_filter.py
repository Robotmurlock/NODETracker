"""
Trajectory forecaster interface definition
"""
import torch
from typing import Optional, Tuple, Protocol


class BBoxTrajectoryForecaster(Protocol):
    """
    Defines interface for bbox trajectory forecasting
    """
    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError('Forward method not implemented!')

    def __call__(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError('Forward method not implemented!')  # Added in order to supress 'Not Callable' warning
