"""
Custom model: RNN-ODE
Like ODE-RNN but without ODE in encoder
"""
from typing import Optional, Tuple

import torch
from torch import nn

from nodetracker.node.core.odevae import NODEDecoder
from nodetracker.node.odernn.utils import LightningGaussianModel, run_simple_lightning_guassian_model_test
from nodetracker.node.utils import LightningTrainConfig
from nodetracker.standard.rnn.seq_to_seq import RNNEncoder


class RNNODE(nn.Module):
    """
    RNN-ODE
    """
    def __init__(
        self,
        observable_dim: int,
        hidden_dim: int,

        model_gaussian: bool = False,

        n_encoder_rnn_layers: int = 1,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None
    ):
        super().__init__()

        self._encoder = RNNEncoder(
            input_dim=observable_dim,  # time is additional obs dimension
            hidden_dim=hidden_dim,
            latent_dim=hidden_dim,
            rnn_n_layers=n_encoder_rnn_layers
        )
        self._decoder = NODEDecoder(
            latent_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=observable_dim,
            solver_name=solver_name,
            solver_params=solver_params,
            model_gaussian=model_gaussian
        )

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        z0 = self._encoder(x, t_obs)
        assert z0.shape[0] == 1, f'Expected temporal dimension to be equal to 1 but got: {z0.shape}!'
        z0 = z0[0]  # Removing temporal dim
        x_hat, z_hat = self._decoder(z0, t_unobs)
        return x_hat, z_hat


class LightningRNNODE(LightningGaussianModel):
    """
    PytorchLightning wrapper for RNNODE model.
    This model is similar to ODERNN but does not use ODE solver in encoder.
    """
    def __init__(
        self,
        observable_dim: int,
        hidden_dim: int,

        model_gaussian: bool = False,

        n_encoder_rnn_layers: int = 1,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None,

        train_config: Optional[LightningTrainConfig] = None
    ):
        model = RNNODE(
            observable_dim=observable_dim,
            hidden_dim=hidden_dim,
            model_gaussian=model_gaussian,
            solver_name=solver_name,
            solver_params=solver_params,
            n_encoder_rnn_layers=n_encoder_rnn_layers
        )
        super().__init__(
            train_config=train_config,
            model=model,
            model_gaussian=model_gaussian
        )

        self._is_modeling_gaussian = model_gaussian

if __name__ == '__main__':
    run_simple_lightning_guassian_model_test(
        model_class=LightningRNNODE,
        params={
            'observable_dim': 7,
            'hidden_dim': 4
        }
    )
