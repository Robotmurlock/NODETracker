"""
Autoregressive RNN
"""
from typing import Optional, Tuple

import torch
from torch import nn

from nodetracker.library import time_series
from nodetracker.library.building_blocks import MLP, ResnetMLPBlock
from nodetracker.node.utils import LightningModuleForecaster, LightningTrainConfig


class ARRNN(nn.Module):
    """
    ARRNN is an autoregressive model that supports variable input sequence and variable output sequence length.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,

        resnet: bool = False,
        resnet_n_layers: int = 2,
        resnet_n_bottleneck_layers: int = 4,

        stem_n_layers: int = 2,
        head_n_layers: int = 1,
        rnn_n_layers: int = 1
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._stem = MLP(
            input_dim=input_dim + 1,  # time dim
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_layers=stem_n_layers
        )
        self._resnet_block = ResnetMLPBlock(
            dim=hidden_dim,
            n_layers=resnet_n_layers,
            n_bottleneck_layers=resnet_n_bottleneck_layers
        ) if resnet else None
        self._rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=rnn_n_layers, batch_first=False)
        self._head = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            n_layers=head_n_layers
        )

    def preprocess(self, x_obs: torch.Tensor, t_obs: torch.Tensor) -> torch.Tensor:
        """
        Preprocess raw trajectory by transforming it into "hidden trajectory" combined with time step differences.

        Args:
            x_obs: Trajectory
            t_obs: Time points

        Returns:
            Hidden trajectory
        """
        # Using relative time point values instead of absolute time point values
        t_diff = time_series.transform.first_difference(t_obs)

        # Add relative time points to input data
        xt = torch.cat([x_obs, t_diff], dim=-1)

        return self._stem(xt)

    def step(self, x_i: torch.Tensor, prev_h: Optional[torch.Tensor] = None):
        """
        Performs single step for history data input.

        Args:
            x_i: Trajectory data point
            prev_h: RNN Hidden state

        Returns:
            Predicted trajectory point, Updated hidden state
        """
        batch_size = x_i.shape[1]

        prev_h = torch.zeros(1, batch_size, self._hidden_dim).to(x_i) if prev_h is None else prev_h
        if self._resnet_block is not None:
            x_i = self._resnet_block(x_i)
        z, prev_h = self._rnn(x_i, prev_h.detach())

        return z, prev_h

    def postprocess(self, zs: torch.Tensor) -> torch.Tensor:
        """
        Projects hidden states to observable space.

        Args:
            zs: Hidden states

        Returns:
            Prediction
        """
        return self._head(zs)

    def forward(self, x_obs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        obs_time_len, unobs_time_len = t_obs.shape[0], t_unobs.shape[0]
        assert obs_time_len >= 1, f'Minimum history length is 1 but found {obs_time_len}.'

        xt = self.preprocess(x_obs, t_obs)

        # WarmUp
        prev_h = None
        z = None
        for i in range(obs_time_len):
            z, prev_h = self.step(xt[i:i+1], prev_h)

        # Autoregressive
        zs = [z[0]]
        for _ in range(unobs_time_len-1):
            z, prev_h = self.step(z, prev_h)
            zs.append(z[0])

        zs = torch.stack(zs)
        x_hats = self.postprocess(zs)

        return x_hats, zs


class LightningARRNN(LightningModuleForecaster):
    """
    Simple RNN implementation to compare with NODE models.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,

        resnet: bool = False,
        resnet_n_layers: int = 2,
        resnet_n_bottleneck_layers: int = 4,

        stem_n_layers: int = 2,
        head_n_layers: int = 1,
        rnn_n_layers: int = 1,

        train_config: Optional[LightningTrainConfig] = None
    ):
        model = ARRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            resnet=resnet,
            resnet_n_layers=resnet_n_layers,
            resnet_n_bottleneck_layers=resnet_n_bottleneck_layers,
            stem_n_layers=stem_n_layers,
            head_n_layers=head_n_layers,
            rnn_n_layers=rnn_n_layers
        )
        loss_func = nn.MSELoss()
        super().__init__(train_config=train_config, model=model, loss_func=loss_func)


def run_test() -> None:
    # Test ARRNN (with and without resnet block)
    xs = torch.randn(4, 3, 7)
    ts_obs = torch.randn(4, 3, 1)
    ts_unobs = torch.randn(2, 3, 1)
    expected_shape = (2, 3, 7)

    for resnet in [False, True]:
        model = LightningARRNN(
            input_dim=7,
            hidden_dim=5,
            resnet=resnet
        )

        output, *_ = model(xs, ts_obs, ts_unobs)
        assert output.shape == expected_shape, f'Expected shape {expected_shape} but found {output.shape}!'
        print(f'Number of parameters (resnet={resnet}): {model.n_params}.')


if __name__ == '__main__':
    run_test()
