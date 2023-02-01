"""
Implementation of simple RNN Encoder Decoder architecture used for comparison
- Encoder: using Same RNNEncoder as in ODEVAE
-
"""
import torch
from torch import nn

from nodetracker.library import time_series
from nodetracker.node.building_blocks import MLP
from nodetracker.node.utils import LightningTrainConfig, LightningModuleBase
from typing import Tuple, Optional
from nodetracker.node.odernn.odernn import LightningModuleRNN


class RNNEncoder(nn.Module):
    """
    Time-series RNN encoder. Can work with time-series with variable lengths and possible missing values.
    Simplified version of NODE latent model RNN encoder.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self._latent_dim = latent_dim
        self._rnn = nn.GRU(input_dim + 1, hidden_dim)
        self._hidden2latent = nn.Linear(hidden_dim, latent_dim)  # outputs log_var and mean for each input

    def forward(self, x_obs: torch.Tensor, t_obs: torch.Tensor) -> torch.Tensor:
        # Using relative time point values instead of absolute time point values
        t_diff = time_series.transform.first_difference(t_obs)

        # Add relative time points to input data
        xt = torch.cat([x_obs, t_diff], dim=-1)

        _, h = self._rnn(xt)
        z = self._hidden2latent(h)
        return z

class RNNDecoder(nn.Module):
    """
    Simple RNN decoder
    """
    def __init__(self, observable_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        self._latent_dim = latent_dim
        self._rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=1,
            batch_first=False
        )
        self._latent2hidden = MLP(input_dim=latent_dim, output_dim=hidden_dim)
        self._hidden2obs = MLP(input_dim=hidden_dim, output_dim=observable_dim)

    def forward(self, z: torch.Tensor, n_steps: int) -> torch.Tensor:
        batch_size = z.shape[1]

        prev_h = torch.zeros(1, batch_size, self._latent_dim, dtype=torch.float32)
        outputs = []
        for _ in range(n_steps):
            z_out, prev_h = self._rnn(z, prev_h)
            h = self._latent2hidden(z_out)
            o = self._hidden2obs(h)
            outputs.append(o)

        return torch.vstack(outputs)


class RNNSeq2Seq(nn.Module):
    """
    Time-series forecasting (sequence to sequence). Supports:
    - variable input sequence length;
    - variable output sequence length;
    - potential missing values (uses time diff)
    """
    def __init__(self, observable_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self._encoder = RNNEncoder(
            input_dim=observable_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        self._decoder = RNNDecoder(
            observable_dim=observable_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )

    def forward(self, x: torch.Tensor, ts_obs: torch.Tensor, ts_unobs: torch.Tensor) -> torch.Tensor:
        n_steps = ts_unobs.shape[0]
        z = self._encoder(x, ts_obs)
        x_hat = self._decoder(z, n_steps)
        return x_hat


class LightningRNNSeq2Seq(LightningModuleRNN):
    """
    Simple RNN implementation to compare with NODE models.
    """
    def __init__(
            self,
            observable_dim: int,
            hidden_dim: int,
            latent_dim: int,

            train_config: Optional[LightningTrainConfig] = None
    ):
        super().__init__(train_config=train_config)
        self._model = RNNSeq2Seq(observable_dim, hidden_dim, latent_dim)
        self._loss_func = nn.MSELoss()


def main() -> None:
    model = LightningRNNSeq2Seq(
        observable_dim=5,
        hidden_dim=3,
        latent_dim=2
    )
    xs = torch.randn(10, 3, 5)
    ts_obs = torch.randn(10, 3, 1)
    ts_unobs = torch.rand(20, 3, 1)
    output = model(xs, ts_obs, ts_unobs)
    expected_shape = (20, 3, 5)

    assert output.shape == expected_shape, f'Expected shape {expected_shape} but found {output.shape}!'


if __name__ == '__main__':
    main()
