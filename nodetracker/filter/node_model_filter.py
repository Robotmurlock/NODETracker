"""
Implementation of NODEKalmanFilter - Gaussian Model.
"""
from typing import Tuple

import torch

from nodetracker.datasets import transforms
from nodetracker.filter.base import StateModelFilter, State
from nodetracker.filter.utils import ODETorchTensorBuffer
from nodetracker.node import LightningNODEFilterModel, extract_mean_and_std


class NODEModelFilter(StateModelFilter):
    """
    Wraps NODE filter into a StateModel (Gaussian filter).
    """
    def __init__(
        self,
        model: LightningNODEFilterModel,
        transform: transforms.InvertibleTransformWithVariance,
        accelerator: str,

        buffer_size: int,
        buffer_min_size: int = 1,
        dtype: torch.dtype = torch.float32
    ):
        self._model = model
        self._transform = transform

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
        assert n_steps == 1, 'Multistep prediction is not supported'

        if not self._buffer.has_input:
            raise BufferError('Buffer does not have input!')

        x_obs, ts_obs, ts_unobs = self._buffer.get_input(n_steps)
        if ts_obs.shape[0] == 1:
            # Only one bbox in history - using baseline (propagate last bbox) instead of NN model
            x_unobs_mean_hat = torch.stack([x_obs[-1].clone() for _ in range(ts_unobs.shape[0])]).to(ts_obs)
            x_unobs_std_hat = 10 * torch.ones_like(x_unobs_mean_hat).to(ts_obs)
            z0, _ = self._model.core.initialize(batch_size=1, device=ts_obs.device)
            z0 = z0.to(self._accelerator)
            return x_unobs_mean_hat[:, 0, :], x_unobs_std_hat[:, 0, :], z0

        t_x_obs, _, t_ts_obs, t_ts_unobs, *_ = self._transform.apply(data=[x_obs, None, ts_obs, ts_unobs, None], shallow=False)
        t_x_obs, t_ts_obs, t_ts_unobs = t_x_obs.to(self._accelerator), t_ts_obs.to(self._accelerator), t_ts_unobs.to(
            self._accelerator)
        z0, _ = self._model.core.encode_obs_trajectory(t_x_obs, t_ts_obs)
        x_prior, z_prior = self._model.core.estimate_prior(z0, t_ts_obs[-1], t_ts_unobs[0])
        t_prior_mean, t_prior_std = extract_mean_and_std(x_prior)
        t_prior_mean, t_prior_std = t_prior_mean.detach().cpu(), t_prior_std.detach().cpu()
        _, prior_mean, *_ = self._transform.inverse(data=[x_obs, t_prior_mean.view(1, 1, -1), None], shallow=False)
        prior_std = self._transform.inverse_std(t_prior_std, additional_data=[x_obs, None], shallow=False)

        prior_mean = prior_mean[:, 0, :]

        return prior_mean, prior_std, z_prior

    def predict(self, state: State) -> State:
        prior_mean, prior_std, z_prior = self.multistep_predict(state, n_steps=1)
        return prior_mean[0], prior_std[0], z_prior

    def update(self, state: State, measurement: torch.Tensor) -> State:
        x_unobs_mean_hat, x_unobs_std_hat, z_prior = state
        x_obs, ts_obs, ts_unobs = self._buffer.get_input(n_future_steps=1)  # Required for transformations
        _, t_measurement, *_ = self._transform.apply(data=[x_obs, measurement.view(1, 1, -1), ts_obs, ts_unobs, None], shallow=False)
        t_measurement = t_measurement[0].to(self._accelerator)
        z_evidence = self._model.core.encode_unobs(t_measurement)
        t_x_posterior, z_posterior = self._model.core.estimate_posterior(z_prior, z_evidence)
        t_x_posterior = t_x_posterior.detach().cpu()
        t_posterior_mean, t_x_posterior_std = extract_mean_and_std(t_x_posterior)
        _, posterior_mean, *_ = self._transform.inverse(data=[x_obs, t_posterior_mean.view(1, 1, -1), None], shallow=False)
        posterior_std = self._transform.inverse_std(t_x_posterior_std, additional_data=[posterior_mean, None], shallow=False)
        self._buffer.push(measurement)

        posterior_mean = posterior_mean[0, 0, :]
        posterior_std = posterior_std[0, :]

        return posterior_mean, posterior_std, z_posterior

    def missing(self, state: State) -> State:
        self._buffer.increment()
        return state

    def project(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std, _ = state
        var = torch.square(std)
        return mean, var


def run_test() -> None:
    from nodetracker.datasets.transforms.base import IdentityTransform

    smf = NODEModelFilter(
        model=LightningNODEFilterModel(
            observable_dim=4,
            latent_dim=4,
            model_gaussian=True,
        ),
        accelerator='cpu',
        buffer_size=10,
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
