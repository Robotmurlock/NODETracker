"""
Implementation of NODEFilter - Gaussian Model.
"""
from dataclasses import dataclass
from typing import Tuple, Optional, List

import torch

from nodetracker.datasets import transforms
from nodetracker.filter.base import StateModelFilter, State
from nodetracker.node import LightningNODEFilterModel
from nodetracker.filter.utils import ODETorchTensorBuffer

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
        assert n_steps == 1, 'Multistep prediction is not supported!'

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
        t_obs, t_unobs, z0 = \
            t_obs.to(self._accelerator), t_unobs.to(self._accelerator), z0.to(self._accelerator)
        t_prior_mean, t_prior_log_var, z_prior = self._model.core.estimate_prior(z0, t_obs, t_unobs)
        t_prior_mean, t_prior_log_var, z_prior = \
            t_prior_mean.detach().cpu(), t_prior_log_var.detach().cpu(), z_prior.detach().cpu()
        t_prior_std = torch.sqrt(self._model.core.postprocess_log_var(t_prior_log_var))
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

    def singlestep_to_multistep_state(self, state: State) -> State:
        state.mean = state.mean.unsqueeze(0)
        state.std = state.std.unsqueeze(0)
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
        z_prior = z_prior.to(self._accelerator)
        t_posterior_mean, t_posterior_log_var, z_posterior = self._model.core.estimate_posterior(z_prior, z_evidence)
        t_posterior_mean, t_posterior_log_var, z_posterior = \
            t_posterior_mean.detach().cpu(), t_posterior_log_var.detach().cpu(), z_posterior.detach().cpu()
        t_posterior_std = torch.sqrt(self._model.core.postprocess_log_var(t_posterior_log_var))
        _, posterior_mean, *_ = self._transform.inverse(data=[x_obs, t_posterior_mean.view(1, 1, -1), None], shallow=False)
        posterior_std = self._transform.inverse_std(t_posterior_std, additional_data=[posterior_mean, None], shallow=False)

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


class BufferedNodeModelFilter(StateModelFilter):
    def __init__(
        self,
        model: LightningNODEFilterModel,
        transform: transforms.InvertibleTransformWithVariance,
        accelerator: str,

        buffer_size: int,
        buffer_min_size: int = 1,
        buffer_min_history: int = 5,
        dtype: torch.dtype = torch.float32,

        recursive_inverse: bool = False
    ):
        self._model = model
        self._transform = transform
        self._accelerator = accelerator
        self._dtype = dtype

        self._buffer_size = buffer_size
        self._buffer_min_size = buffer_min_size
        self._buffer_min_history = buffer_min_history

        # Model
        self._model.to(self._accelerator)
        self._model.eval()

        self._recursive_inverse = recursive_inverse

    def _get_ts(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t_obs = torch.tensor([0], dtype=self._dtype).view(1, 1)
        t_unobs = torch.tensor([1], dtype=self._dtype).view(1, 1)
        return t_obs, t_unobs

    def initiate(self, measurement: torch.Tensor) -> State:
        buffer = ODETorchTensorBuffer(
            size=self._buffer_size,
            min_size=self._buffer_min_size,
            min_history=self._buffer_min_history,
            dtype=self._dtype
        )
        buffer.push(measurement)
        return buffer, None, None, None

    def multistep_predict(self, state: State, n_steps: int) -> State:
        assert n_steps == 1, 'Multistep prediction is not supported!'

        buffer, _, _, _ = state
        if not buffer.has_input:
            raise BufferError('Buffer does not have an input!')

        x_obs, ts_obs, ts_unobs = buffer.get_input(n_steps)
        if ts_obs.shape[0] == 1:
            # Only one bbox in history - using baseline (propagate last bbox) instead of NN model
            x_unobs_mean_hat = torch.stack([x_obs[-1].clone() for _ in range(ts_unobs.shape[0])]).to(ts_obs)
            x_unobs_std_hat = 10 * torch.ones_like(x_unobs_mean_hat).to(ts_obs)

            z0, _ = self._model.core.initialize(batch_size=1, device=self._accelerator)
            t_obs, t_unobs = self._get_ts()
            t_obs, t_unobs = t_obs.to(self._accelerator), t_unobs.to(self._accelerator)
            _, _, z1_prior = self._model.core.estimate_prior(z0, t_obs, t_unobs)
            z1_prior = z1_prior.detach().cpu()

            return buffer, x_unobs_mean_hat[:, 0, :], x_unobs_std_hat[:, 0, :], z1_prior

        ts_first = ts_obs[0, 0, 0]
        ts_obs = ts_obs - ts_first + 1
        ts_unobs = ts_unobs - ts_first + 1
        t_obs_last, t_unobs_last = ts_obs[-1, 0, 0], ts_unobs[-1, 0, 0]
        if self._recursive_inverse:
            ts_unobs = torch.tensor(list(range(int(t_obs_last) + 1, int(t_unobs_last) + 1)), dtype=torch.float32).view(-1, 1, 1)

        t_x_obs, _, t_ts_obs, t_ts_unobs, *_ = self._transform.apply(data=[x_obs, None, ts_obs, ts_unobs, None], shallow=False)
        t_x_obs, t_ts_obs, t_ts_unobs = t_x_obs.to(self._accelerator), t_ts_obs.to(self._accelerator), t_ts_unobs.to(
            self._accelerator)
        x_unobs_dummy = torch.zeros(1, 1, 4).to(self._accelerator)
        t_x_prior_mean, t_x_prior_var, _, _, z_prior = \
            self._model.forward(t_x_obs, t_ts_obs, x_unobs_dummy, t_ts_unobs, mask=[False for _ in range(ts_unobs.shape[0])])
        x_unobs_dummy.detach().cpu()
        t_x_prior_mean = t_x_prior_mean.detach().cpu()
        t_x_prior_mean = t_x_prior_mean.detach().cpu()
        z_prior = z_prior.detach().cpu()

        _, prior_mean, *_ = self._transform.inverse(data=[x_obs, t_x_prior_mean, None], shallow=False)

        prior_mean = prior_mean[-1]  # Removing temporal dimension after inverse transform
        prior_var = self._transform.inverse_var(t_x_prior_var, additional_data=[x_obs, None], shallow=False)
        prior_std = torch.sqrt(prior_var)[0]

        return buffer, prior_mean, prior_std, z_prior

    def predict(self, state: State) -> State:
        buffer, prior_mean, prior_std, z1_prior = self.multistep_predict(state, n_steps=1)
        return buffer, prior_mean[0], prior_std[0], z1_prior

    def singlestep_to_multistep_state(self, state: State) -> State:
        buffer, prior_mean, prior_std, z1_prior = state
        return buffer, prior_mean.unsqueeze(0), prior_std.unsqueeze(0), z1_prior

    def update(self, state: State, measurement: torch.Tensor) -> State:
        buffer, _, _, z_prior = state
        x_obs, ts_obs, ts_unobs = buffer.get_input(1)

        if ts_obs.shape[0] == 1:
            # Only one bbox in history - using baseline (propagate last bbox) instead of NN model
            x_unobs_mean_hat = torch.stack([x_obs[-1].clone() for _ in range(ts_unobs.shape[0])]).to(ts_obs)
            x_unobs_std_hat = 10 * torch.ones_like(x_unobs_mean_hat).to(ts_obs)

            z0, _ = self._model.core.initialize(batch_size=1, device=self._accelerator)
            t_obs, t_unobs = self._get_ts()
            t_obs, t_unobs = t_obs.to(self._accelerator), t_unobs.to(self._accelerator)
            _, _, z1_prior = self._model.core.estimate_prior(z0, t_obs, t_unobs)
            z1_prior = z1_prior.detach().cpu()

            buffer.push(measurement)
            return buffer, x_unobs_mean_hat[0, 0, :], x_unobs_std_hat[0, 0, :], z1_prior

        _, t_measurement, *_ = self._transform.apply(data=[x_obs, measurement.view(1, 1, -1), ts_obs, ts_unobs, None], shallow=False)
        t_measurement, z_prior = t_measurement[0].to(self._accelerator), z_prior.to(self._accelerator)
        z_evidence = self._model.core.encode_unobs(t_measurement)
        t_posterior_mean, t_posterior_log_var, z_posterior = self._model.core.estimate_posterior(z_prior, z_evidence)
        t_posterior_mean, t_posterior_log_var, z_posterior = \
            t_posterior_mean.detach().cpu(), t_posterior_log_var.detach().cpu(), z_posterior.detach().cpu()
        t_posterior_var = self._model.core.postprocess_log_var(t_posterior_log_var)
        _, posterior_mean, *_ = self._transform.inverse(data=[x_obs, t_posterior_mean.view(1, 1, -1), None], shallow=False)
        posterior_var = self._transform.inverse_var(t_posterior_var, additional_data=[posterior_mean, None], shallow=False)
        posterior_std = torch.sqrt(posterior_var)

        posterior_mean = posterior_mean[0, 0, :]
        posterior_std = posterior_std[0, :]

        buffer.push(measurement)

        return buffer, posterior_mean, posterior_std, z_posterior

    def missing(self, state: State) -> State:
        buffer, mean, std, z_state = state
        buffer.increment()
        return buffer, mean, std, z_state

    def project(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        _, mean, std, _ = state
        var = torch.square(std)
        return mean, var


def run_test() -> None:
    from nodetracker.datasets.transforms.base import IdentityTransform

    smf = NODEModelFilter(
        model=LightningNODEFilterModel(
            observable_dim=4,
            latent_dim=4,
        ),
        accelerator='cpu',
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
