"""
Autoregressive single-step RNN
"""
from typing import Optional, Tuple, Union

import torch
from torch import nn

from nodetracker.datasets.transforms import InvertibleTransform, InvertibleTransformWithVariance
from nodetracker.library import time_series
from nodetracker.library.building_blocks import MLP
from nodetracker.node.odernn.utils import LightningGaussianModel
from nodetracker.node.utils import LightningTrainConfig


class SingleStepRNN(nn.Module):
    """
    SingleStepRNN supports variable length input sequence. Output is always single-step.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,

        model_gaussian: bool = False,

        stem_n_layers: int = 2,
        head_n_layers: int = 1,
        rnn_n_layers: int = 1,

        rnn_dropout: float = 0.3
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._stem = MLP(
            input_dim=input_dim + 1,  # time dim
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_layers=stem_n_layers
        )
        self._rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=rnn_n_layers, batch_first=False)
        self._dropout = nn.Dropout(rnn_dropout)

        self._head = nn.Sequential(
            MLP(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                n_layers=head_n_layers
            ),
            nn.Linear(hidden_dim, output_dim if not model_gaussian else 2 * output_dim, bias=True)
        )

    def forward(self, x_obs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        _ = t_unobs  # Unused

        obs_time_len = t_obs.shape[0]
        assert obs_time_len >= 1, f'Minimum history length is 1 but found {obs_time_len}.'

        # Using relative time point values instead of absolute time point values
        t_diff = time_series.transform.first_difference(t_obs)

        # Add relative time points to input data
        xt = torch.cat([x_obs, t_diff], dim=-1)

        xt = self._stem(xt)
        z, _ = self._rnn(xt)
        z = self._dropout(z)
        out = self._head(z[-1]).unsqueeze(0)  # Add time step dimension (1, ...)
        return out


class LightningSingleStepRNN(LightningGaussianModel):
    """
    Simple RNN implementation to compare with NODE models.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,

        stem_n_layers: int = 2,
        head_n_layers: int = 1,
        rnn_n_layers: int = 1,

        model_gaussian: bool = False,
        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None,

        train_config: Optional[LightningTrainConfig] = None,
        log_epoch_metrics: bool = True
    ):
        if output_dim is None:
            output_dim = input_dim

        model = SingleStepRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            stem_n_layers=stem_n_layers,
            head_n_layers=head_n_layers,
            rnn_n_layers=rnn_n_layers,
            model_gaussian=model_gaussian
        )
        super().__init__(
            train_config=train_config,
            model=model,
            model_gaussian=model_gaussian,
            transform_func=transform_func,
            log_epoch_metrics=log_epoch_metrics
        )


def run_test() -> None:
    xs = torch.randn(4, 3, 7)
    ts_obs = torch.randn(4, 3, 1)
    ts_unobs = torch.randn(2, 3, 1)
    expected_shape = (1, 3, 4)

    model = SingleStepRNN(
        input_dim=7,
        hidden_dim=5,
        output_dim=4
    )

    output = model(xs, ts_obs, ts_unobs)
    assert output.shape == expected_shape, f'Expected shape {expected_shape} but found {output.shape}!'


if __name__ == '__main__':
    run_test()
