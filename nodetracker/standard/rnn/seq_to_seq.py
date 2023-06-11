"""
Implementation of simple RNN Encoder Decoder architecture used for comparison
- Encoder: using same form of RNNEncoder as in ODERNN
- Decoder: Variable step regression
"""
from typing import Optional, Union

import torch
from torch import nn

from nodetracker.datasets.transforms import InvertibleTransform, InvertibleTransformWithVariance
from nodetracker.library import time_series
from nodetracker.library.building_blocks import MLP
from nodetracker.node.odernn.utils import LightningGaussianModel, run_simple_lightning_guassian_model_test
from nodetracker.node.utils import LightningTrainConfig


class RNNEncoder(nn.Module):
    """
    Time-series RNN encoder. Can work with time-series with variable lengths and possible missing values.
    Simplified version of NODE latent model RNN encoder.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,

        rnn_n_layers: int = 1
    ):
        """
        Args:
            input_dim: Trajectory point dimension
            hidden_dim: Hidden trajectory dimension
            latent_dim: "latent" (hidden-2) trajectory dimension
            rnn_n_layers: Number of stacked RNN (GRU) layers
        """
        super().__init__()
        self._latent_dim = latent_dim
        self._rnn = nn.GRU(input_dim + 1, hidden_dim, num_layers=rnn_n_layers, batch_first=False)
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
    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        output_dim: int,

        model_gaussian: bool = False,

        rnn_n_layers: int = 1
    ):
        """
        Args:
            hidden_dim: Hidden trajectory dimension
            latent_dim: "latent" (hidden-2) trajectory dimension
            rnn_n_layers: Number of stacked RNN (GRU) layers
        """
        super().__init__()

        self._latent_dim = latent_dim
        self._rnn_n_layers = rnn_n_layers

        self._rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=rnn_n_layers,
            batch_first=False
        )
        self._latent2hidden = MLP(input_dim=latent_dim, output_dim=hidden_dim)
        self._hidden2obs = nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim if not model_gaussian else 2 * output_dim,
            bias=True
        )

    def forward(self, z: torch.Tensor, n_steps: int) -> torch.Tensor:
        batch_size = z.shape[1]

        prev_h = torch.zeros(self._rnn_n_layers, batch_size, self._latent_dim, dtype=torch.float32).to(z).requires_grad_()
        outputs = []
        for _ in range(n_steps):
            z_out, prev_h = self._rnn(z, prev_h.detach())
            z_out = z_out[-1]
            h = self._latent2hidden(z_out)
            o = self._hidden2obs(h)
            outputs.append(o)

        return torch.stack(outputs)


class RNNSeq2Seq(nn.Module):
    """
    Time-series forecasting (sequence to sequence). Supports:
    - variable input sequence length;
    - variable output sequence length;
    - potential missing values (uses time diff)
    """
    def __init__(
        self,
        observable_dim: int,
        hidden_dim: int,
        latent_dim: int,
        output_dim: int,

        model_gaussian: bool = False,

        encoder_rnn_n_layers: int = 1,
        decoder_rnn_n_layers: int = 1
    ):
        """
        Args:
            observable_dim: Trajectory point dimension
            hidden_dim: Hidden trajectory dimension
            latent_dim: "latent" (hidden-2) trajectory dimension
            encoder_rnn_n_layers: Number of stacked RNN (GRU) layers for RNN Encoder
            decoder_rnn_n_layers: Number of stacked RNN (GRU) layers for RNN Decoder
        """
        super().__init__()
        self._encoder = RNNEncoder(
            input_dim=observable_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,

            rnn_n_layers=encoder_rnn_n_layers
        )
        self._decoder = RNNDecoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,

            model_gaussian=model_gaussian,

            rnn_n_layers=decoder_rnn_n_layers
        )

    def forward(self, x: torch.Tensor, ts_obs: torch.Tensor, ts_unobs: torch.Tensor) -> torch.Tensor:
        n_steps = ts_unobs.shape[0]
        z = self._encoder(x, ts_obs)
        x_hat = self._decoder(z, n_steps)
        return x_hat


class LightningRNNSeq2Seq(LightningGaussianModel):
    """
    Simple RNN implementation to compare with NODE models.
    """
    def __init__(
            self,
            observable_dim: int,
            hidden_dim: int,
            latent_dim: int,

            output_dim: Optional[int] = None,
            encoder_rnn_n_layers: int = 1,
            decoder_rnn_n_layers: int = 1,

            model_gaussian: bool = False,
            transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None,

            train_config: Optional[LightningTrainConfig] = None,
            log_epoch_metrics: bool = True
    ):
        """
        Args:
            observable_dim: Trajectory point dimension
            hidden_dim: Hidden trajectory dimension
            latent_dim: "latent" (hidden-2) trajectory dimension
            encoder_rnn_n_layers: Number of stacked RNN (GRU) layers for RNN Encoder
            decoder_rnn_n_layers: Number of stacked RNN (GRU) layers for RNN Decoder
        """
        if output_dim is None:
            output_dim = observable_dim

        model = RNNSeq2Seq(
            observable_dim=observable_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,

            model_gaussian=model_gaussian,

            encoder_rnn_n_layers=encoder_rnn_n_layers,
            decoder_rnn_n_layers=decoder_rnn_n_layers
        )
        super().__init__(
            train_config=train_config,
            model=model,
            model_gaussian=model_gaussian,
            transform_func=transform_func,
            log_epoch_metrics=log_epoch_metrics
        )


def main() -> None:
    run_simple_lightning_guassian_model_test(
        model_class=LightningRNNSeq2Seq,
        params={
            'observable_dim': 7,
            'hidden_dim': 3,
            'latent_dim': 2,

            'encoder_rnn_n_layers': 2,
            'decoder_rnn_n_layers': 2
        }
    )


if __name__ == '__main__':
    main()
