"""
Variational Autoencoder loss function - ELBO
"""
from typing import Tuple

import torch
from torch import nn


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
