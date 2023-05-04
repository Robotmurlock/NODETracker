from typing import Tuple, List

import numpy as np
import torch

from nodetracker.filter.base import State
from nodetracker.filter.base import StateModelFilter
from nodetracker.library.kalman_filter.botsort_kf import BotSortKalmanFilter


class BotSortKalmanFilterWrapper(StateModelFilter):
    """
    Wrapper for BotSortKalman filter for StateModelFilter interface.
    """
    def __init__(self, use_optimal_motion_mat: bool = False):
        self._kf = BotSortKalmanFilter(use_optimal_motion_mat=use_optimal_motion_mat)

    @staticmethod
    def _all_to_numpy(state: State) -> Tuple[np.ndarray, ...]:
        """
        Converts all tensors in state to numpy arrays.

        Args:
            state: Current state

        Returns:
            all tensors converted as numpy arrays.
        """
        return tuple(s.numpy() for s in state)

    @staticmethod
    def _all_to_tensor(state: List[np.ndarray]) -> State:
        """
        Converts all numpy arrays to tensor state (on proper device).

        Args:
            state: Current state as numpy arrays

        Returns:
            all numpy arrays converted as tensors.
        """
        return tuple(torch.tensor(s) for s in state)

    def initiate(self, measurement: torch.Tensor) -> State:
        mean, covariance = self._kf.initiate(measurement)
        mean, covariance = self._all_to_tensor([mean, covariance])
        return mean, covariance

    def predict(self, state: State) -> State:
        mean, covariance = self._all_to_numpy(state)
        mean_hat, covariance_hat = self._kf.predict(mean, covariance)
        mean_hat, covariance_hat = self._all_to_tensor([mean_hat, covariance_hat])
        return mean_hat, covariance_hat

    def update(self, state: State, measurement: torch.Tensor) -> State:
        mean_hat, covariance_hat = self._all_to_numpy(state)
        mean, covariance = self._kf.update(mean_hat, covariance_hat, measurement)
        mean, covariance = self._all_to_tensor([mean, covariance])
        return mean, covariance

    def project(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, covariance = self._all_to_numpy(state)
        mean_proj, covariance_proj = self._kf.project(mean, covariance)
        mean_proj, covariance_proj = self._all_to_tensor([mean_proj, covariance_proj])
        return mean_proj, covariance_proj