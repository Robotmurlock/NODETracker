from typing import Tuple

import torch

from nodetracker.datasets import transforms
from nodetracker.filter.base import StateModelFilter, State
from nodetracker.filter.utils import ODETorchTensorBuffer
from nodetracker.node.odernn.utils import extract_mean_and_var
from nodetracker.np import LightningRNNCNPFilter


class CNPFilter(StateModelFilter):
    def __init__(
        self,
        model: LightningRNNCNPFilter,
        transform: transforms.InvertibleTransformWithVariance,
        accelerator: str,

        buffer_size: int,
        buffer_min_size: int = 1,
        dtype: torch.dtype = torch.float32
    ):
        self._model = model
        self._transform = transform
        self._accelerator = accelerator
        self._dtype = dtype

        # Buffer
        self._buffer = ODETorchTensorBuffer(
            size=buffer_size,
            min_size=buffer_min_size,
            dtype=dtype
        )

        # Model
        self._model.to(self._accelerator)
        self._model.eval()

    def _get_ts(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t_obs = torch.tensor([0], dtype=self._dtype).view(1, 1)
        t_unobs = torch.tensor([1], dtype=self._dtype).view(1, 1)
        return t_obs, t_unobs

    def initiate(self, measurement: torch.Tensor) -> State:
        self._buffer.push(measurement)
        return None

    @staticmethod
    def baseline(
        x_obs: torch.Tensor,
        ts_unobs: torch.Tensor,
        single_step: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        x_unobs_mean_hat = torch.stack([x_obs[-1].clone() for _ in range(ts_unobs.shape[0])]).to(x_obs)
        x_unobs_std_hat = 10 * torch.ones_like(x_unobs_mean_hat).to(x_obs)
        if single_step:
            return x_unobs_mean_hat[0, 0, :], x_unobs_std_hat[0, 0, :], None
        return x_unobs_mean_hat[:, 0, :], x_unobs_std_hat[:, 0, :], None

    def multistep_predict(self, state: State, n_steps: int) -> State:
        assert n_steps == 1, 'Multistep prediction is not supported!'

        if not self._buffer.has_input:
            raise BufferError('Buffer does not have an input!')

        x_obs, ts_obs, ts_unobs = self._buffer.get_input(n_steps)
        if ts_obs.shape[0] == 1:
            # Only one bbox in history - using baseline (propagate last bbox) instead of NN model
            return self.baseline(x_obs, ts_unobs, single_step=False)

        t_x_obs, _, t_ts_obs, t_ts_unobs, *_ = self._transform.apply(data=[x_obs, None, ts_obs, ts_unobs, None], shallow=False)
        t_x_obs, t_ts_obs, t_ts_unobs = t_x_obs.to(self._accelerator), t_ts_obs.to(self._accelerator), t_ts_unobs.to(
            self._accelerator)
        t_prior, xhr2 = self._model.core.estimate_prior(t_ts_obs, t_x_obs, t_ts_unobs)
        t_prior_mean, t_prior_var = extract_mean_and_var(t_prior)
        t_prior_mean, t_prior_var = t_prior_mean.detach().cpu(), t_prior_var.detach().cpu()
        _, prior_mean, *_ = self._transform.inverse(data=[x_obs, t_prior_mean, None], shallow=False)
        prior_var = self._transform.inverse_var(t_prior_var, additional_data=[x_obs, None], shallow=False)
        prior_std = torch.sqrt(prior_var)

        prior_mean, prior_std = prior_mean[:, 0, :], prior_std[:, 0, :]

        return prior_mean, prior_std, xhr2

    def predict(self, state: State) -> State:
        prior_mean, prior_std, xhr2 = self.multistep_predict(state, n_steps=1)
        prior_mean, prior_std = prior_mean[0], prior_std[0]
        return prior_mean, prior_std, xhr2

    def singlestep_to_multistep_state(self, state: State) -> State:
        prior_mean, prior_std, xhr2 = state
        prior_mean, prior_std = prior_mean.unsqueeze(0), prior_std.unsqueeze(0)
        return prior_mean, prior_std, xhr2

    def update(self, state: State, measurement: torch.Tensor) -> State:
        _, _, xhr2 = state
        x_obs, ts_obs, ts_unobs = self._buffer.get_input(1)

        if xhr2 is None:
            self._buffer.push(measurement)

            # Only one bbox in history - using baseline (propagate last bbox) instead of NN model
            return self.baseline(x_obs, ts_unobs, single_step=True)

        _, t_measurement, *_ = self._transform.apply(data=[x_obs, measurement.view(1, 1, -1), ts_obs, ts_unobs, None], shallow=False)
        xhr2, t_measurement = xhr2.to(self._accelerator), t_measurement.to(self._accelerator)
        t_posterior = self._model.core.estimate_posterior(xhr2, t_measurement)
        t_posterior = t_posterior.detach().cpu()
        t_posterior_mean, t_posterior_var = extract_mean_and_var(t_posterior)
        _, posterior_mean, *_ = self._transform.inverse(data=[x_obs, t_posterior_mean.view(1, 1, -1), None], shallow=False)
        posterior_var = self._transform.inverse_var(t_posterior_var, additional_data=[posterior_mean, None], shallow=False)
        posterior_std = torch.sqrt(posterior_var)

        posterior_mean = posterior_mean[0, 0, :]
        posterior_std = posterior_std[0, 0, :]

        self._buffer.push(measurement)

        return posterior_mean, posterior_std, None

    def missing(self, state: State) -> State:
        self._buffer.increment()
        return state

    def project(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std, _ = state
        var = torch.square(std)
        return mean, var