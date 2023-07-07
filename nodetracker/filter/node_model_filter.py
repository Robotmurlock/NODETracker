"""
Implementation of NODEKalmanFilter - Gaussian Model.
"""
from dataclasses import dataclass
from typing import Tuple, Optional

import torch

from nodetracker.datasets import transforms
from nodetracker.filter.base import StateModelFilter, State
from nodetracker.node import LightningNODEFilterModel, extract_mean_and_std

Estimation = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class NODEState:
    measurement: torch.Tensor
    z: torch.Tensor
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    baseline: bool = True
    missing_counter: int = 0

    def copy(self):
        return NODEState(
            measurement=self.measurement.clone(),
            z=self.z.clone(),
            mean=self.mean.clone() if self.mean is not None else None,
            std=self.std.clone() if self.mean is not None else None,
            baseline=self.baseline,
            missing_counter=self.missing_counter
        )


class NODEModelFilter(StateModelFilter):
    """
    Wraps NODE filter into a StateModel (Gaussian filter).
    """
    def __init__(
        self,
        model: LightningNODEFilterModel,
        transform: transforms.InvertibleTransformWithVariance,
        accelerator: str,

        dtype: torch.dtype = torch.float32
    ):
        self._model = model
        self._transform = transform
        self._accelerator = accelerator
        self._dtype = dtype

        self._model.to(self._accelerator)
        self._model.eval()

    def initiate(self, measurement: torch.Tensor) -> State:
        z, _ = self._model.core.initialize(batch_size=1, device=self._accelerator)
        return NODEState(measurement=measurement, z=z)

    def _get_ts(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t_obs = torch.tensor([0], dtype=self._dtype).view(1, 1)
        t_unobs = torch.tensor([1], dtype=self._dtype).view(1, 1)
        return t_obs, t_unobs

    def multistep_predict(self, state: State, n_steps: int) -> State:
        state: NODEState
        state = state.copy()

        if state.baseline:
            # Using baseline (propagate last bbox) instead of NN model
            last_measurement = state.measurement.unsqueeze(0)
            x_unobs_mean_hat = torch.stack([last_measurement.clone() for _ in range(n_steps)]).to(last_measurement)
            x_unobs_std_hat = 10 * torch.ones_like(x_unobs_mean_hat).to(last_measurement)

            # Update state
            state.mean = x_unobs_mean_hat[:, 0, :]
            state.std = x_unobs_std_hat[:, 0, :]

            return state

        z0 = state.z
        t_obs, t_unobs = self._get_ts()
        t_obs, t_unobs = t_obs.to(self._accelerator), t_unobs.to(self._accelerator)
        x_prior, z_prior = self._model.core.estimate_prior(z0, t_obs, t_unobs)
        t_prior_mean, t_prior_std = extract_mean_and_std(x_prior)
        t_prior_mean, t_prior_std = t_prior_mean.detach().cpu(), t_prior_std.detach().cpu()
        x_obs = state.measurement.unsqueeze(0)
        _, prior_mean, *_ = self._transform.inverse(data=[x_obs, t_prior_mean.view(1, 1, -1), None], shallow=False)
        prior_std = self._transform.inverse_std(t_prior_std, additional_data=[x_obs, None], shallow=False)

        # Update state
        state.z = z_prior
        state.mean = prior_mean[:, 0, :]
        state.std = prior_std

        return state

    def predict(self, state: State) -> State:
        state = self.multistep_predict(state, n_steps=1)
        state.mean, state.std = state.mean[0], state.std[0]
        return state

    def update(self, state: State, measurement: torch.Tensor) -> State:
        state: NODEState
        state = state.copy()

        z_prior = state.z
        x_obs = state.measurement.unsqueeze(0)
        t_obs, t_unobs = self._get_ts()

        _, t_measurement, *_ = self._transform.apply(data=[x_obs, measurement.view(1, 1, -1), t_obs, t_unobs, None], shallow=False)
        t_measurement = t_measurement[0].to(self._accelerator)
        z_evidence = self._model.core.encode_unobs(t_measurement)
        t_x_posterior, z_posterior = self._model.core.estimate_posterior(z_prior, z_evidence)
        t_x_posterior = t_x_posterior.detach().cpu()
        t_posterior_mean, t_x_posterior_std = extract_mean_and_std(t_x_posterior)
        _, posterior_mean, *_ = self._transform.inverse(data=[x_obs, t_posterior_mean.view(1, 1, -1), None], shallow=False)
        posterior_std = self._transform.inverse_std(t_x_posterior_std, additional_data=[posterior_mean, None], shallow=False)

        posterior_mean = posterior_mean[0, 0, :]
        posterior_std = posterior_std[0, :]

        # Update state
        state.measurement = measurement
        state.z = z_posterior
        state.mean = posterior_mean
        state.std = posterior_std
        state.baseline = False

        return state

    def missing(self, state: State) -> State:
        state: NODEState
        state = state.copy()

        # Update state
        # FIXME: should update measurement!!!!
        state.missing_counter += 1
        if state.missing_counter >= 10:
            state.baseline = True

        return self.predict(state)

    def project(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        state: NODEState
        mean, std = state.mean, state.std
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
