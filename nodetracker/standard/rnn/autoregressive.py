"""
Autoregressive RNN
"""
from typing import Optional, Tuple, Union

import torch
from torch import nn

from nodetracker.library import time_series
from nodetracker.library.building_blocks import MLP, ResnetMLPBlock
from nodetracker.datasets.transforms import InvertibleTransform, InvertibleTransformWithVariance
from nodetracker.node.utils import LightningModuleForecasterWithTeacherForcing, LightningTrainConfig


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

        rnn_dropout: float = 0.3,

        stem_n_layers: int = 2,
        head_n_layers: int = 1,
        rnn_n_layers: int = 1
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._rnn_n_layers = rnn_n_layers

        self._stem = MLP(
            input_dim=input_dim + 1,  # time dim
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            n_layers=stem_n_layers
        )
        self._resnet_block: Optional[ResnetMLPBlock] = ResnetMLPBlock(
            dim=input_dim,
            n_layers=resnet_n_layers,
            n_bottleneck_layers=resnet_n_bottleneck_layers
        ) if resnet else None
        self._rnn = nn.GRU(input_dim, hidden_dim, num_layers=rnn_n_layers, batch_first=False)
        self._dropout = nn.Dropout(rnn_dropout)
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

        prev_h = torch.zeros(self._rnn_n_layers, batch_size, self._hidden_dim).to(x_i) if prev_h is None else prev_h
        if self._resnet_block is not None:
            x_i = self._resnet_block(x_i)
        z, prev_h = self._rnn(x_i, prev_h.detach())
        z = self._dropout(z)
        x_hat = self._head(z)

        return x_hat, prev_h

    def forward(self, x_obs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor, x_tf: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor]:
        obs_time_len, unobs_time_len = t_obs.shape[0], t_unobs.shape[0]
        assert obs_time_len >= 1, f'Minimum history length is 1 but found {obs_time_len}.'

        xt = self.preprocess(x_obs, t_obs)

        # WarmUp
        prev_h = None
        x_hat = None
        for i in range(obs_time_len):
            x_hat, prev_h = self.step(xt[i:i+1], prev_h)


        # Autoregressive
        x_hats = [x_hat]
        for i in range(unobs_time_len-1):
            x_hat = x_hat if x_tf is None else x_tf[i:i+1]
            x_hat, prev_h = self.step(x_hat, prev_h)
            x_hats.append(x_hat)

        x_hats = torch.cat(x_hats)
        # noinspection PyTypeChecker
        return x_hats,


class LightningARRNN(LightningModuleForecasterWithTeacherForcing):
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

        model_gaussian: bool = False,
        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None,
        teacher_forcing: bool = False,

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
        super().__init__(
            train_config=train_config,
            model=model,
            model_gaussian=model_gaussian,
            transform_func=transform_func,
            teacher_forcing=teacher_forcing
        )


def run_test() -> None:
    # Test ARRNN (with and without resnet block)
    xs = torch.randn(4, 3, 7)
    ts_obs = torch.randn(4, 3, 1)
    ts_unobs = torch.randn(2, 3, 1)
    x_unobs = torch.randn(2, 3, 7)
    expected_shape = (2, 3, 7)

    for teacher_forcing in [False, True]:
        for resnet in [False, True]:
            model = LightningARRNN(
                input_dim=7,
                hidden_dim=5,
                resnet=resnet,
                teacher_forcing=teacher_forcing
            )

            x_tf = x_unobs if teacher_forcing else None
            output, *_ = model(xs, ts_obs, ts_unobs, x_tf=x_tf)
            assert output.shape == expected_shape, f'Expected shape {expected_shape} but found {output.shape}!'
            print(f'Number of parameters (resnet={resnet}): {model.n_params}.')


if __name__ == '__main__':
    run_test()
