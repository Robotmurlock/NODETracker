from typing import Tuple, List

import torch

from nodetracker.filter.base import StateModelFilter, State
from nodetracker.datasets import transforms
from nodetracker.node import LightningGaussianModel


class ODETorchTensorBuffer:
    def __init__(self, size: int, min_size: int, dtype: torch.dtype):
        assert size >= 1, f'Invalid size {size}. Minimum size is 1.'

        self._size = size
        self._min_size = min_size
        self._dtype = dtype

        self._buffer: List[torch.Tensor] = []

    @property
    def has_input(self) -> bool:
        return len(self._buffer) >= self._min_size

    def push(self, x: torch.Tensor) -> None:
        self._buffer.append(x)
        if len(self._buffer) > self._size:
            self._buffer.pop(0)

    def get_input(self, n_future_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_hist_steps = len(self._buffer)
        x_obs = torch.stack(self._buffer).view(n_hist_steps, 1, -1)
        ts_obs = torch.tensor(list(range(1, n_hist_steps + 1)), dtype=torch.float32).view(-1, 1, 1)
        ts_unobs = torch.tensor(list(range(n_hist_steps + 1, n_hist_steps + n_future_steps + 1)), dtype=torch.float32).view(-1, 1, 1)

        return x_obs, ts_obs, ts_unobs

    def clear(self) -> None:
        self._buffer.clear()

class NODEFilter(StateModelFilter):
    """
    Wraps NODE filter into a StateModel (Gaussian filter).
    """
    def __init__(
        self,
        model: LightningGaussianModel,
        transform: transforms.InvertibleTransformWithStd,
        accelerator: str,
        det_std: List[float],

        n_pred_steps: int,

        buffer_size: int,
        buffer_min_size: int,
        dtype: torch.dtype = torch.float32
    ):
        self._model = model
        self._transform = transform
        self._n_pred_steps = n_pred_steps
        self._det_std = torch.tensor(det_std, dtype=dtype)

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

    def predict(self, state: State) -> State:
        if not self._buffer.has_input:
            raise BufferError('Buffer does not have input!')

        x_obs, ts_obs, ts_unobs = self._buffer.get_input(self._n_pred_steps)
        if ts_obs.shape[0] == 1:
            # Only one bbox in history - using baseline (propagate last bbox) instead of NN model
            x_unobs_mean_hat = torch.stack([x_obs[-1].clone() for _ in range(ts_unobs.shape[0])]).to(ts_obs)
            x_unobs_std_hat = 10 * torch.ones_like(x_unobs_mean_hat).to(ts_obs)
            return x_unobs_mean_hat, x_unobs_std_hat

        t_x_obs, _, t_ts_obs, *_ = self._transform.apply(data=(x_obs, None, ts_obs), shallow=False)
        t_x_obs, t_ts_obs, ts_unobs = t_x_obs.to(self._accelerator), t_ts_obs.to(self._accelerator), ts_unobs.to(self._accelerator)
        t_x_unobs_mean_hat, t_x_unobs_std_hat, *_ = self._model.inference(t_x_obs, t_ts_obs, ts_unobs)
        t_x_unobs_mean_hat, t_x_unobs_std_hat = t_x_unobs_mean_hat.detach().cpu(), t_x_unobs_std_hat.detach().cpu()
        _, prior_mean, *_ = self._transform.inverse(data=[x_obs, t_x_unobs_mean_hat], shallow=True)
        prior_std = self._transform.inverse_std(t_x_unobs_std_hat)

        return prior_mean, prior_std

    def update(self, state: State, measurement: torch.Tensor) -> State:
        x_unobs_mean_hat, x_unobs_std_hat = state
        det_std = self._det_std.to(x_unobs_mean_hat)

        def std_inv(m: torch.Tensor) -> torch.Tensor:
            return (1 / m).nan_to_num(nan=0, posinf=0, neginf=0)

        x_unobs_cov_hat_inv = std_inv(x_unobs_std_hat)
        det_cov_inv = std_inv(det_std)
        gain = std_inv(x_unobs_cov_hat_inv + det_cov_inv) * det_cov_inv

        innovation = (measurement - x_unobs_mean_hat)
        posterior_mean = x_unobs_mean_hat + gain * innovation
        posterior_std = (1 - gain) * x_unobs_std_hat

        return posterior_mean, posterior_std

    def project(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = state
        return mean, std
