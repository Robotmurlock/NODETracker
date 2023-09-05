"""
Autoregressive single-step RNN with optical flow.
"""
from typing import Optional, Tuple, Union

import torch
from torch import nn

from nodetracker.datasets.transforms import InvertibleTransform, InvertibleTransformWithVariance
from nodetracker.library import time_series
from nodetracker.library.building_blocks import MLP
from nodetracker.node.odernn.utils import LightningGaussianModel
from nodetracker.node.utils import LightningTrainConfig
from nodetracker.library.building_blocks import CNNBlock


class SingleStepFlowRNN(nn.Module):
    """
    SingleStepFlowRNN supports variable length input sequence. Output is always single-step.
    Also encoded camera flow (motion) for trajectory prediction
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

        rnn_dropout: float = 0.3,

        cnn_channels: int = 2
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._stem = MLP(
            input_dim=input_dim + 1,  # time dim
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_layers=stem_n_layers
        )
        self._flow_cnn = nn.Sequential(
            CNNBlock(
                in_channels=cnn_channels,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=1
            ),
            CNNBlock(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            CNNBlock(
                in_channels=64,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim
            ),
            nn.ReLU()
        )
        self._merger = nn.Sequential(
            nn.Linear(
                in_features=2 * hidden_dim,
                out_features=hidden_dim
            ),
            nn.ReLU()
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

    def forward(self, x_obs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor, metadata: dict) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        _ = t_unobs  # Unused
        obs_time_steps, batch_size, _ = x_obs.shape

        obs_time_len = t_obs.shape[0]
        assert obs_time_len >= 1, f'Minimum history length is 1 but found {obs_time_len}.'

        # Using relative time point values instead of absolute time point values
        t_diff = time_series.transform.first_difference(t_obs)

        # Add relative time points to input data
        xt = torch.cat([x_obs, t_diff], dim=-1)

        zt = self._stem(xt)
        flow = metadata['flow'][1:, :, :, :, :]
        flow = flow.reshape(-1, *flow.shape[-3:])
        z_flow = self._flow_cnn(flow)
        z_flow = z_flow.reshape(obs_time_steps, batch_size, -1)
        z_all = torch.cat([zt, z_flow], dim=-1)
        z_all = self._merger(z_all)

        z, _ = self._rnn(z_all)
        z = self._dropout(z)
        out = self._head(z[-1]).unsqueeze(0)  # Add time step dimension (1, ...)
        return out


class LightningSingleStepFlowRNN(LightningGaussianModel):
    """
    Wrapper for `SingleStepFlowRNN`.
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
        log_epoch_metrics: bool = True,

        cnn_channels: int = 2
    ):
        if output_dim is None:
            output_dim = input_dim

        model = SingleStepFlowRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            stem_n_layers=stem_n_layers,
            head_n_layers=head_n_layers,
            rnn_n_layers=rnn_n_layers,
            model_gaussian=model_gaussian,
            cnn_channels=cnn_channels
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
    flow = torch.randn(5, 3, 2, 224, 224)
    metadata = {'flow': flow}
    ts_obs = torch.randn(4, 3, 1)
    ts_unobs = torch.randn(2, 3, 1)
    expected_shape = (1, 3, 4)

    model = SingleStepFlowRNN(
        input_dim=7,
        hidden_dim=5,
        output_dim=4
    )

    output = model(xs, ts_obs, ts_unobs, metadata)
    assert output.shape == expected_shape, f'Expected shape {expected_shape} but found {output.shape}!'


if __name__ == '__main__':
    run_test()
