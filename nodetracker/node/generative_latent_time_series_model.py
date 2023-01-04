"""
Generative latent functions time-series model
"""
from typing import Tuple, Optional

import pytorch_lightning as pl
import torch
from torch import nn

from nodetracker.library import time_series
from nodetracker.node.core import ODEF, NeuralODE
from nodetracker.utils.meter import MetricMeter


class RNNEncoder(nn.Module):
    """
    Time-series RNN encoder. Can work with time-series with variable lengths and possible missing values.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self._latent_dim = latent_dim

        self._rnn = nn.GRU(input_dim + 1, hidden_dim)
        self._lrelu = nn.LeakyReLU(0.1)
        self._hidden2latent = nn.Linear(hidden_dim, 2 * latent_dim)  # outputs log_var and mean for each input

    def forward(self, x_obs: torch.Tensor, t_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Using relative time point values instead of absolute time point values
        t_diff = time_series.transform.first_difference(t_obs)

        # Add relative time points to input data
        xt = torch.cat([x_obs, t_diff], dim=-1)

        # Reverse input time sequence (running RNN backward in time)
        xt = torch.flip(xt, dims=[0])

        _, h0 = self._rnn(xt)
        h0 = self._lrelu(h0[0])
        z0 = self._hidden2latent(h0)

        z0_mean, z0_log_var = z0[:, :self._latent_dim], z0[:, self._latent_dim:]
        return z0_mean, z0_log_var


class MLPODEF(ODEF):
    """
    Multi layer perceptron ODEF. Includes N consecutive layers of:
    - Linear layer
    - LayerNorm
    - LeakyReLU
    """
    def __init__(self, dim: int, hidden_dim: int, n_layers: int = 2):
        super(ODEF, self).__init__()
        assert n_layers >= 1, f'Minimum number of layers is 1 but found {n_layers}'
        layers_args = [[hidden_dim, hidden_dim] for _ in range(n_layers)]
        layers_args[0][0] = dim + 1  # setting input dimension (including time dimension)
        layers_args[-1][1] = dim  # setting output dimension

        self._model = nn.Sequential(*[self._create_mlp_layer(*args) for args in layers_args])

    @staticmethod
    def _create_mlp_layer(input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt = torch.cat([x, t], dim=-1)
        return self._model(xt)


class NODEDecoder(nn.Module):
    """
    NODE decoder performs extrapolation at latent space which can be then used to reconstruct/forecast time-series.
    """
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self._ode = NeuralODE(MLPODEF(latent_dim, hidden_dim, n_layers=2))
        self._latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self._lrelu = nn.LeakyReLU(0.1)
        self._hidden2output = nn.Linear(hidden_dim, output_dim)

    def forward(self, z0: torch.Tensor, ts: torch.Tensor):
        ts = ts - ts[0, 0, 0] + 1  # Transform to relative values [1, 2, ...]
        zs = self._ode(z0, ts, full_sequence=True)
        hs = self._lrelu(self._latent2hidden(zs))
        xs = self._lrelu(self._hidden2output(hs))
        return xs


class ODEVAE(nn.Module):
    """
    ODEVAE - A generative latent function time-series model
    """
    def __init__(self, observable_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        self._encoder = RNNEncoder(observable_dim, hidden_dim, latent_dim)
        self._decoder = NODEDecoder(latent_dim, hidden_dim, observable_dim)

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_all: Optional[torch.Tensor] = None, generate: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_all = t_all if t_all is not None else t_obs
        assert t_all.shape[0] >= t_obs.shape[0], \
            f'All time points must contain at lease observable time points. Shapes (all, obs): {t_all.shape[0]}, {t_obs.shape[0]}'

        z0_mean, z0_log_var = self._encoder(x, t_obs)
        z0 = z0_mean if not generate else z0_mean + torch.randn_like(z0_mean) * torch.exp(0.5 * z0_log_var)
        x_hat = self._decoder(z0, t_all)
        return x_hat, z0_mean, z0_log_var


class ELBO(nn.Module):
    """
    Implementation of VAE ELBO loss function
    """
    def __init__(self, noise_std: float):
        super().__init__()
        self._noise_std = noise_std

    def forward(self, x_hat: torch.Tensor, x_true: torch.Tensor, z0_mean: torch.Tensor, z0_log_var: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kl_div_loss = torch.sum(1 + z0_log_var - torch.square(z0_mean) - torch.exp(z0_log_var), dim=-1) / 2
        likelihood_loss = torch.mean(torch.sum(torch.pow(x_hat - x_true, 2) / (2 * self._noise_std), dim=-1),
                                     dim=0)  # summation over last dimension and mean over time dimension
        batch_loss = likelihood_loss - kl_div_loss
        return torch.mean(batch_loss), torch.mean(kl_div_loss), torch.mean(likelihood_loss)


class LightningODEVAE(pl.LightningModule):
    """
    PytorchLightning wrapper for ODEVAE model
    """
    def __init__(self, observable_dim: int, hidden_dim: int, latent_dim: int, noise_std: float = 0.1, learning_rate: float = 1e-3):
        super().__init__()
        self._model = ODEVAE(observable_dim, hidden_dim, latent_dim)
        self._loss_func = ELBO(noise_std)
        self._learning_rate = learning_rate

        self._meter = MetricMeter()

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_all: Optional[torch.Tensor] = None, generate: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._model(x, t_obs, t_all, generate)

    def training_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, _, ts_obs, _ = batch
        bboxes_hat, z0_mean, z0_log_var = self.forward(bboxes_obs, ts_obs)
        loss, kl_div_loss, likelihood_loss = self._loss_func(bboxes_hat, bboxes_obs, z0_mean, z0_log_var)

        self._meter.push('train/loss', loss)
        self._meter.push('train/kl_div_loss', kl_div_loss)
        self._meter.push('train/likelihood_loss', likelihood_loss)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs = batch
        ts_all = torch.cat([ts_obs, ts_unobs], dim=0)
        bboxes_all = torch.cat([bboxes_obs, bboxes_unobs], dim=0)

        bboxes_hat, z0_mean, z0_log_var = self.forward(bboxes_obs, ts_obs, ts_all)
        loss, kl_div_loss, likelihood_loss = self._loss_func(bboxes_hat, bboxes_all, z0_mean, z0_log_var)

        self._meter.push('val/loss', loss)
        self._meter.push('val/kl_div_loss', kl_div_loss)
        self._meter.push('val/likelihood_loss', likelihood_loss)

        return loss

    def on_validation_epoch_end(self) -> None:
        for name, value in self._meter.get_all():
            self.log(name, value, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self._model.parameters(), lr=self._learning_rate)
        return [optimizer]


def run_test():
    model = LightningODEVAE(4, 3, 2)
    x_obs = torch.randn(3, 1, 4)
    t_obs = torch.randn(3, 1, 1)

    print('Output:', [x.shape for x in model(x_obs, t_obs)])


if __name__ == '__main__':
    run_test()
