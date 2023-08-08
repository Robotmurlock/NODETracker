from typing import Tuple, List

import torch


class ODETorchTensorBuffer:
    def __init__(self, size: int, min_size: int, dtype: torch.dtype):
        assert size >= 1, f'Invalid size {size}. Minimum size is 1.'

        self._size = size
        self._min_size = min_size
        self._dtype = dtype

        self._buffer: List[Tuple[int, torch.Tensor]] = []
        self._t = 0

    @property
    def time(self) -> int:
        return self._t

    @property
    def has_input(self) -> bool:
        return len(self._buffer) >= self._min_size

    def push(self, x: torch.Tensor) -> None:
        self._buffer.append((self._t, x))
        self.increment()

    def increment(self) -> None:
        if len(self._buffer) > 1 and (self._t - self._buffer[0][0]) >= self._size:
            self._buffer.pop(0)

        self._t += 1

    def get_input(self, n_future_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_hist_steps = len(self._buffer)
        ts_obs, x_obs = zip(*self._buffer)
        ts_obs_first = ts_obs[0]

        # Form history trajectory
        x_obs = torch.stack(x_obs).view(n_hist_steps, 1, -1)
        ts_obs = torch.tensor(ts_obs, dtype=self._dtype).view(-1, 1, 1)

        # Form estimation trajectory time interval
        ts_unobs = torch.tensor(list(range(self._t, self._t + n_future_steps)),
                                dtype=self._dtype).view(-1, 1, 1)

        ts_obs = ts_obs - ts_obs_first + 1
        ts_unobs = ts_unobs - ts_obs_first + 1

        return x_obs, ts_obs, ts_unobs

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)