"""
PytorchLighting Trainer for ODEVAE and ODERNNVAE
"""
from typing import Tuple, Optional

import pytorch_lightning as pl
import torch
from torch import nn

from nodetracker.node.generative_latent_time_series_model import ODEVAE
from nodetracker.node.ode_rnn import ODERNNVAE
from nodetracker.utils.meter import MetricMeter


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
    def __init__(self, model_name: str, observable_dim: int, hidden_dim: int, latent_dim: int, noise_std: float = 0.1, learning_rate: float = 1e-3):
        super().__init__()
        if model_name == 'odevae':
            model_cls = ODEVAE
        elif model_name == 'odernnvae':
            model_cls = ODERNNVAE
        else:
            raise ValueError('Invalid model name. Expected: "odevae" or "odernnvae"!')
        self._model = model_cls(observable_dim, hidden_dim, latent_dim)
        self._loss_func = ELBO(noise_std)
        self._learning_rate = learning_rate

        self._meter = MetricMeter()

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None, generate: bool = False) \
            -> Tuple[torch.Tensor, ...]:
        return self._model(x, t_obs, t_unobs, generate)

    @staticmethod
    def preprocess_batch(batch: tuple) -> tuple:
        """
        Unpacks batch and creates full trajectory tensors

        Args:
            batch: Raw batch

        Returns:
            Preprocessing batch
        """
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata = batch
        ts_all = torch.cat([ts_obs, ts_unobs], dim=0)
        bboxes_all = torch.cat([bboxes_obs, bboxes_unobs], dim=0)
        return bboxes_obs, bboxes_unobs, bboxes_all, ts_obs, ts_unobs, ts_all, metadata

    def training_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, _, bboxes_all, ts_obs, ts_unobs, _, _ = self.preprocess_batch(batch)

        _, bboxes_hat_all, z0_mean, z0_log_var = self.forward(bboxes_obs, ts_obs, ts_unobs)
        loss, kl_div_loss, likelihood_loss = self._loss_func(bboxes_hat_all, bboxes_all, z0_mean, z0_log_var)

        self._meter.push('training/loss', loss)
        self._meter.push('training/kl_div_loss', kl_div_loss)
        self._meter.push('training/likelihood_loss', likelihood_loss)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_unobs, bboxes_all, ts_obs, ts_unobs, _, _ = self.preprocess_batch(batch)

        bboxes_unobs_hat, bboxes_all_hat, z0_mean, z0_log_var = self.forward(bboxes_obs, ts_obs, ts_unobs)
        all_loss, all_kl_div_loss, all_likelihood_loss = self._loss_func(bboxes_all_hat, bboxes_all, z0_mean, z0_log_var)
        unobs_loss, unobs_kl_div_loss, unobs_likelihood_loss = self._loss_func(bboxes_unobs_hat, bboxes_unobs, z0_mean, z0_log_var)

        self._meter.push('val/loss', all_loss)
        self._meter.push('val/kl_div_loss', all_kl_div_loss)
        self._meter.push('val/likelihood_loss', all_likelihood_loss)

        self._meter.push('val-forecast/loss', unobs_loss)
        self._meter.push('val-forecast/kl_div_loss', unobs_kl_div_loss)
        self._meter.push('val-forecast/likelihood_loss', unobs_likelihood_loss)

        return all_loss

    def on_validation_epoch_end(self) -> None:
        for name, value in self._meter.get_all():
            self.log(name, value, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self._model.parameters(), lr=self._learning_rate)
        return [optimizer]
