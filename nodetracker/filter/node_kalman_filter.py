"""
Implementation of NODEKalmanFilter - Gaussian Model.
"""
from typing import Tuple

import torch

from nodetracker.datasets import transforms
from nodetracker.filter.base import StateModelFilter, State
from nodetracker.filter.utils import ODETorchTensorBuffer
from nodetracker.node import LightningGaussianModel


class NODEKalmanFilter(StateModelFilter):
    """
    Wraps NODE filter into a StateModel (Gaussian filter).
    """
    def __init__(
        self,
        model: LightningGaussianModel,
        transform: transforms.InvertibleTransformWithVariance,
        accelerator: str,
        det_uncertainty_multiplier: float,

        buffer_size: int,
        buffer_min_size: int = 1,
        dtype: torch.dtype = torch.float32
    ):
        self._model = model
        self._transform = transform
        self._det_uncertainty_multiplier = det_uncertainty_multiplier

        self._accelerator = accelerator
        self._model.to(self._accelerator)
        self._model.eval()

        self._buffer = ODETorchTensorBuffer(
            size=buffer_size,
            min_size=buffer_min_size,
            dtype=dtype
        )

    def initiate(self, measurement: torch.Tensor) -> State:
        self._buffer.push(measurement)
        return None

    def multistep_predict(self, state: State, n_steps: int) -> State:
        if not self._buffer.has_input:
            raise BufferError('Buffer does not have an input!')

        x_obs, ts_obs, ts_unobs = self._buffer.get_input(n_steps)
        if ts_obs.shape[0] == 1:
            # Only one bbox in history - using baseline (propagate last bbox) instead of NN model
            x_unobs_mean_hat = torch.stack([x_obs[-1].clone() for _ in range(ts_unobs.shape[0])]).to(ts_obs)
            x_unobs_std_hat = 10 * torch.ones_like(x_unobs_mean_hat).to(ts_obs)
            return x_unobs_mean_hat[:, 0, :], x_unobs_std_hat[:, 0, :]

        t_x_obs, _, t_ts_obs, t_ts_unobs, *_ = self._transform.apply(data=[x_obs, None, ts_obs, ts_unobs, None], shallow=False)
        t_x_obs, t_ts_obs, t_ts_unobs = t_x_obs.to(self._accelerator), t_ts_obs.to(self._accelerator), t_ts_unobs.to(
            self._accelerator)
        t_x_unobs_mean_hat, t_x_unobs_std_hat, *_ = self._model.inference(t_x_obs, t_ts_obs, t_ts_unobs)
        t_x_unobs_mean_hat, t_x_unobs_std_hat = t_x_unobs_mean_hat.detach().cpu(), t_x_unobs_std_hat.detach().cpu()
        _, prior_mean, *_ = self._transform.inverse(data=[x_obs, t_x_unobs_mean_hat, None], shallow=False)
        prior_std = self._transform.inverse_std(t_x_unobs_std_hat, additional_data=[x_obs, None], shallow=False)

        prior_mean = prior_mean[:, 0, :]
        prior_std = prior_std[:, 0, :]

        return prior_mean, prior_std

    def predict(self, state: State) -> State:
        prior_mean, prior_std = self.multistep_predict(state, n_steps=1)
        return prior_mean[0], prior_std[0]

    def update(self, state: State, measurement: torch.Tensor) -> State:
        x_unobs_mean_hat, x_unobs_std_hat = state
        det_std = (x_unobs_mean_hat[2:].repeat(2) * self._det_uncertainty_multiplier)

        def std_inv(m: torch.Tensor) -> torch.Tensor:
            return (1 / m).nan_to_num(nan=0, posinf=0, neginf=0)

        x_unobs_cov_hat_inv = std_inv(x_unobs_std_hat)
        det_cov_inv = std_inv(det_std)
        gain = std_inv(x_unobs_cov_hat_inv + det_cov_inv) * det_cov_inv

        innovation = (measurement - x_unobs_mean_hat)
        posterior_mean = x_unobs_mean_hat + gain * innovation
        posterior_std = (1 - gain) * x_unobs_std_hat

        self._buffer.push(measurement)

        return posterior_mean, posterior_std

    def missing(self, state: State) -> State:
        self._buffer.increment()
        return state

    def project(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = state
        var = torch.square(std)
        return mean, var


def run_test() -> None:
    from nodetracker.node.odernn import LightningRNNODE
    from nodetracker.datasets.transforms.base import IdentityTransform

    smf = NODEKalmanFilter(
        model=LightningRNNODE(
            observable_dim=4,
            hidden_dim=4,
            model_gaussian=True,
        ),
        accelerator='cpu',
        buffer_size=10,
        det_uncertainty_multiplier=0.05,
        transform=IdentityTransform()
    )

    measurements = torch.randn(10, 4, dtype=torch.float32)

    # Initiate test
    initial_state = smf.initiate(measurements[0])
    assert initial_state is None

    # Predict test
    mean_hat, cov_hat = smf.predict(initial_state)
    assert mean_hat.shape == (4,) and cov_hat.shape == (4,)

    # Projected predict
    mean_hat_proj, cov_hat_proj = smf.project((mean_hat, cov_hat))
    assert mean_hat_proj.shape == (4,) and cov_hat_proj.shape == (4,)

    # Update test
    mean_updated, cov_updated = smf.update((mean_hat, cov_hat), measurements[1])
    assert mean_updated.shape == (4,) and cov_updated.shape == (4,)

    # Projected update
    mean_updated_proj, cov_updated_proj = smf.project((mean_updated, cov_updated))
    assert mean_updated_proj.shape == (4,) and cov_updated_proj.shape == (4,)

    # Multistep predict
    mean_multistep_hat, cov_multistep_hat = smf.multistep_predict(initial_state, n_steps=5)
    assert mean_multistep_hat.shape == (5, 4) and cov_multistep_hat.shape == (5, 4)

    # Projected multistep
    mean_multistep_hat_proj, cov_multistep_hat_proj = smf.project((mean_multistep_hat, cov_multistep_hat))
    assert mean_multistep_hat_proj.shape == (5, 4) and cov_multistep_hat_proj.shape == (5, 4)


if __name__ == '__main__':
    run_test()
