"""
Kalman Filter torch wrapper.
"""
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from nodetracker.library import time_series
from nodetracker.library.kalman_filter.kalman_filter import ConstantVelocityODKalmanFilter


class TorchConstantVelocityODKalmanFilter(nn.Module):
    """
    Kalman Filter torch wrapper
    """
    def __init__(
        self,
        time_step_multiplier: float = 1.0,
        initial_position_uncertainty: float = 10,
        initial_velocity_uncertainty: float = 10000,
        process_noise_multiplier: float = 1.0,
        measurement_noise_multiplier: float = 1.0
    ):
        """
        Args:
            time_step_multiplier: Time point values multiplier
                - hyperparamater is dependent on fps
            initial_position_uncertainty: Initial position uncertainty
                - higher values -> Kalman gain is more biased towards measurement
            initial_velocity_uncertainty: Similar to `initial_position_uncertainty`
                but for velocity (recommended high values)
            process_noise_multiplier: Increase/decrease process noise
                - higher values -> more KF prediction uncertainty
            measurement_noise_multiplier: Increase/decrease measurement noise
                - higher values -> Kalman gain is less biased towards measurement
        """
        super().__init__()
        self._kf = ConstantVelocityODKalmanFilter(
            initial_position_uncertainty=initial_position_uncertainty,
            initial_velocity_uncertainty=initial_velocity_uncertainty,
            process_noise_multiplier=process_noise_multiplier,
            measurement_noise_multiplier=measurement_noise_multiplier
        )
        self._time_step_multiplier = time_step_multiplier

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, ...]:
        assert len(x.shape) == 3, f'Kalman filter expects (t, 1, 4) shape but found {x.shape}.'
        assert x.shape[1] == 1, f'Kalman filter does not support batch size different than 1! Got shape: {x.shape}.'
        assert x.shape[2] == 4, f'Kalman filter expected 4 measurements! Got shape: {x.shape}.'
        self._kf.reset_state()

        n_obs = t_obs.shape[0]
        n_unobs = t_unobs.shape[0]
        device = x.device

        x = x.detach().cpu().numpy()
        t_obs_diff = time_series.transform.first_difference(t_obs).detach().cpu().numpy()
        t_obs_diff[-1, 0, 0] = 1.0  # Changing first values from 0 to 1
        t_unobs_relative = t_unobs - t_obs[-1, :, :]  # relative to last obs value
        t_unobs_relative = t_unobs_relative.detach().cpu().numpy()

        initial_measurement = x[0, 0, :].reshape(4, 1)
        self._kf.set_z(initial_measurement)
        for i in range(1, n_obs):
            dt = t_obs_diff[i, 0, 0].item() * self._time_step_multiplier
            measurement = x[i, 0, :].reshape(4, 1)

            self._kf.predict(dt)
            self._kf.update(measurement)

        preds = []
        for i in range(n_unobs):
            dt = t_unobs_relative[i, 0, 0].item() * self._time_step_multiplier
            pred, _ = self._kf.predict(dt)
            pred = pred.reshape(1, -1)
            preds.append(pred)

        batched_preds = np.stack(preds)
        batched_preds = torch.from_numpy(batched_preds).to(device)
        return batched_preds,


def main() -> None:
    tkf = TorchConstantVelocityODKalmanFilter(
        process_noise_multiplier=0.1,
        measurement_noise_multiplier=0.1
    )
    x = torch.randn(5, 1, 4)
    t_obs = torch.tensor([10, 11, 12, 13, 14], dtype=torch.float32).view(-1, 1, 1)
    t_unobs = torch.tensor([15, 17, 18], dtype=torch.float32).view(-1, 1, 1)

    output, *_ = tkf(x, t_obs, t_unobs)
    expected_shape = (3, 1, 4)
    assert output.shape == expected_shape, f'Expected shape {expected_shape} but found {output.shape}!'


if __name__ == '__main__':
    main()
