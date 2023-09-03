from typing import Tuple, List, Optional

import numpy as np
import torch

from nodetracker.filter.base import State
from nodetracker.filter.base import StateModelFilter
from nodetracker.library.kalman_filter.botsort_kf import BotSortKalmanFilter


class BotSortKalmanFilterWrapper(StateModelFilter):
    """
    Wrapper for BotSortKalman filter for StateModelFilter interface.
    """
    def __init__(
        self,
        use_optimal_motion_mat: bool = False,
        override_std_weight_position: Optional[float] = None,
        optimal_motion_mat_name: str = 'lasot'
    ):
        self._kf = BotSortKalmanFilter(
            use_optimal_motion_mat=use_optimal_motion_mat,
            override_std_weight_position=override_std_weight_position,
            optimal_motion_mat_name=optimal_motion_mat_name
        )

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

    def multistep_predict(self, state: State, n_steps: int) -> State:
        mean, covariance = state
        mean_hat_list, cov_hat_list = [], []
        for _ in range(n_steps):
            mean, covariance = self.predict((mean, covariance))
            mean_hat_list.append(mean)
            cov_hat_list.append(covariance)

        mean_hat = torch.stack(mean_hat_list)
        covariance_hat = torch.stack(cov_hat_list)
        return mean_hat, covariance_hat

    def update(self, state: State, measurement: torch.Tensor) -> State:
        mean_hat, covariance_hat = self._all_to_numpy(state)
        mean, covariance = self._kf.update(mean_hat, covariance_hat, measurement)
        mean, covariance = self._all_to_tensor([mean, covariance])
        return mean, covariance

    def singlestep_to_multistep_state(self, state: State) -> State:
        mean, covariance = state
        return mean.unsqueeze(0), covariance.unsqueeze(0)

    def missing(self, state: State) -> State:
        return state  # Use prior instead of posterior

    def project(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, covariance = self._all_to_numpy(state)
        if len(mean.shape) == 1:
            # Single step projection
            mean_proj, covariance_proj = self._kf.project(mean, covariance)
        elif len(mean.shape) == 2:
            # Multistep (batch) projection
            batch_size = mean.shape[0]
            mean_proj_list, cov_proj_list = [], []
            for i in range(batch_size):
                mean_i, covariance_i = mean[i, :], covariance[i, :, :]
                mean_proj_i, covariance_proj_i = self._kf.project(mean_i, covariance_i)
                mean_proj_list.append(mean_proj_i)
                cov_proj_list.append(covariance_proj_i)

            mean_proj = np.stack(mean_proj_list)
            covariance_proj = np.stack(cov_proj_list)
        else:
            raise AssertionError(f'Invalid shape {mean.shape}')

        mean_proj, covariance_proj = self._all_to_tensor([mean_proj, covariance_proj])
        return mean_proj, torch.diagonal(covariance_proj, dim1=-2, dim2=-1)


def run_test() -> None:
    smf = BotSortKalmanFilterWrapper()
    measurements = torch.randn(10, 4, dtype=torch.float32)

    # Initiate test
    mean, cov = smf.initiate(measurements[0])
    assert mean.shape == (8,) and cov.shape == (8, 8)

    # Predict test
    mean_hat, cov_hat = smf.predict((mean, cov))
    assert mean_hat.shape == (8,) and cov_hat.shape == (8, 8)

    # Projected predict
    mean_hat_proj, cov_hat_proj = smf.project((mean_hat, cov_hat))
    assert mean_hat_proj.shape == (4,) and cov_hat_proj.shape == (4, 4)

    # Update test
    mean_updated, cov_updated = smf.update((mean, cov), measurements[1])
    assert mean_updated.shape == (8,) and cov_updated.shape == (8, 8)

    # Projected update
    mean_updated_proj, cov_updated_proj = smf.project((mean_updated, cov_updated))
    assert mean_updated_proj.shape == (4,) and cov_updated_proj.shape == (4, 4)

    # Multistep predict
    mean_multistep_hat, cov_multistep_hat = smf.multistep_predict((mean, cov), n_steps=5)
    assert mean_multistep_hat.shape == (5, 8) and cov_multistep_hat.shape == (5, 8, 8)

    # Projected multistep
    mean_multistep_hat_proj, cov_multistep_hat_proj = smf.project((mean_multistep_hat, cov_multistep_hat))
    assert mean_multistep_hat_proj.shape == (5, 4) and cov_multistep_hat_proj.shape == (5, 4, 4)


if __name__ == '__main__':
    run_test()
