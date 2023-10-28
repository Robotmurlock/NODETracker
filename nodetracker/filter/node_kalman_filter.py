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
        buffer_min_history: int = 1,
        dtype: torch.dtype = torch.float32,
        autoregressive: bool = False,
        recursive_inverse: bool = False
    ):
        self._model = model
        self._transform = transform
        self._det_uncertainty_multiplier = det_uncertainty_multiplier

        self._accelerator = accelerator
        self._model.to(self._accelerator)
        self._model.eval()

        self._buffer_size = buffer_size
        self._buffer_min_size = buffer_min_size
        self._buffer_min_history = buffer_min_history
        self._dtype = dtype

        assert not autoregressive or not recursive_inverse
        self._autoregressive = autoregressive
        self._recursive_inverse = recursive_inverse

    def initiate(self, measurement: torch.Tensor) -> State:
        buffer = ODETorchTensorBuffer(
            size=self._buffer_size,
            min_size=self._buffer_min_size,
            min_history=self._buffer_min_history,
            dtype=self._dtype
        )
        buffer.push(measurement)
        return buffer, None, None

    def multistep_predict(self, state: State, n_steps: int) -> State:
        assert n_steps == 1, 'Multistep predict not supported!'

        buffer, _, _ = state
        if not buffer.has_input:
            raise BufferError('Buffer does not have an input!')

        x_obs, ts_obs, ts_unobs = buffer.get_input(n_steps)
        assert ts_obs.shape[1] == 1, 'Batch operations not supported!'

        if ts_obs.shape[0] == 1:
            # Only one bbox in history - using baseline (propagate last bbox) instead of NN model
            x_unobs_mean_hat = torch.stack([x_obs[-1].clone() for _ in range(ts_unobs.shape[0])]).to(ts_obs)
            x_unobs_std_hat = 10 * torch.ones_like(x_unobs_mean_hat).to(ts_obs)
            return buffer, x_unobs_mean_hat[:, 0, :], x_unobs_std_hat[:, 0, :]

        t_obs_last, t_unobs_last = ts_obs[-1, 0, 0], ts_unobs[-1, 0, 0]
        n_ar_steps = 1 if not self._autoregressive else int(t_unobs_last - t_obs_last)
        if self._recursive_inverse:
            ts_unobs = torch.tensor(list(range(int(t_obs_last) + 1, int(t_unobs_last) + 1)), dtype=torch.float32).view(-1, 1, 1)

        for i in range(n_ar_steps):
            t_x_obs, _, t_ts_obs, t_ts_unobs, *_ = self._transform.apply(data=[x_obs, None, ts_obs, ts_unobs, None], shallow=False)
            t_x_obs, t_ts_obs, t_ts_unobs = t_x_obs.to(self._accelerator), t_ts_obs.to(self._accelerator), t_ts_unobs.to(
                self._accelerator)
            t_x_unobs_mean_hat, t_x_unobs_std_hat, *_ = self._model.inference(t_x_obs, t_ts_obs, t_ts_unobs)
            t_x_unobs_mean_hat, t_x_unobs_std_hat = t_x_unobs_mean_hat.detach().cpu(), t_x_unobs_std_hat.detach().cpu()
            _, prior_mean, *_ = self._transform.inverse(data=[x_obs, t_x_unobs_mean_hat, None], shallow=False)
            prior_std = self._transform.inverse_std(t_x_unobs_std_hat, additional_data=[x_obs, None], shallow=False)

            x_obs = torch.cat([x_obs, prior_mean], dim=0)
            t_curr = int(t_obs_last) + i + 1
            ts_obs = torch.cat([ts_obs, torch.tensor([t_curr], dtype=self._dtype).view(1, 1, 1)])

            prior_mean = prior_mean[-1:, 0, :]
            prior_std = prior_std[-1:, 0, :]

        # noinspection PyUnboundLocalVariable
        return buffer, prior_mean, prior_std

    def predict(self, state: State) -> State:
        buffer, prior_mean, prior_std = self.multistep_predict(state, n_steps=1)
        return buffer, prior_mean[0], prior_std[0]

    def singlestep_to_multistep_state(self, state: State) -> State:
        buffer, prior_mean, prior_std = state
        return buffer, prior_mean.unsqueeze(0),  prior_std.unsqueeze(0)

    def update(self, state: State, measurement: torch.Tensor) -> State:
        buffer, x_unobs_mean_hat, x_unobs_std_hat = state
        det_std = (x_unobs_mean_hat[2:].repeat(2) * self._det_uncertainty_multiplier)
        det_cov = torch.square(det_std)
        x_unobs_cov_hat = torch.square(x_unobs_std_hat)

        def cov_inv(m: torch.Tensor) -> torch.Tensor:
            return (1 / m).nan_to_num(nan=0, posinf=0, neginf=0)

        x_unobs_cov_hat_inv = cov_inv(x_unobs_cov_hat)
        det_cov_inv = cov_inv(det_cov)
        gain = cov_inv(x_unobs_cov_hat_inv + det_cov_inv) * det_cov_inv

        innovation = (measurement - x_unobs_mean_hat)
        posterior_mean = x_unobs_mean_hat + gain * innovation
        posterior_cov = (1 - gain) * x_unobs_cov_hat
        posterior_std = torch.sqrt(posterior_cov)

        buffer.push(measurement)

        return buffer, posterior_mean, posterior_std

    def missing(self, state: State) -> State:
        buffer, mean, std = state
        buffer.increment()
        return buffer, mean, std

    def project(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        _, mean, std = state
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
