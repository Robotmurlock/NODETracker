"""
ODE-RNN-VAE implementation
https://arxiv.org/pdf/1907.03907.pdf
"""
from typing import Tuple, Optional

import torch
from torch import nn

from nodetracker.node.building_blocks import MLP
from nodetracker.node.core.original import NeuralODE
from nodetracker.node.core.solver import ode_solver_factory
from nodetracker.node.core.odevae import MLPODEF, NODEDecoder, ELBO
from nodetracker.node.utils import LightningTrainConfig, LightningModuleBase


class ODERNNVAEEncoder(nn.Module):
    """
    ODE-RNN Encoder. Encoder uses combination of NODE and RNN (NODE -> RNN).
    """
    def __init__(
        self,
        observable_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_node_mlp_layers: int = 2,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None
    ):
        """
        Args:
            observable_dim: Data dimension
            latent_dim: Latent dimension (dimension at which NODE performs extrapolation)
            hidden_dim: Hidden dimension (additional dimension between observable and latent)
            n_node_mlp_layers: Number NODE MLP layers
        """
        super().__init__()
        assert n_node_mlp_layers >= 1, f'Minimum number of NODE MLP layers is 1 but found {n_node_mlp_layers}.'

        self._latent_dim = latent_dim
        self._obs2hidden = MLP(input_dim=observable_dim, output_dim=hidden_dim)
        self._hidden2latent = MLP(input_dim=hidden_dim, output_dim=2*latent_dim)  # mean and std

        odef_func = MLPODEF(
            dim=2*latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_node_mlp_layers
        )

        solver = ode_solver_factory(name=solver_name, params=solver_params)
        self._node = NeuralODE(func=odef_func, solver=solver)
        self._rnn = nn.GRU(
            input_size=2*latent_dim,
            hidden_size=2*latent_dim,
            num_layers=1,
            batch_first=False
        )

    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        time_len, batch_size, _ = xs.shape

        hs = self._obs2hidden(xs)
        zs = self._hidden2latent(hs)
        prev_h0 = torch.zeros(1, batch_size, 2*self._latent_dim).to(zs)

        z0 = None
        for i in range(time_len):
            ts = torch.tensor([0, 1], dtype=torch.float32).view(2, 1, 1).expand(2, batch_size, 1).to(zs)
            z0 = zs[i]

            z0 = self._node(z0, ts).unsqueeze(0)
            z0, prev_h0 = self._rnn(z0, prev_h0)

        assert z0 is not None, 'Zero ODERNN iterations performed!'
        assert z0.shape[0] == 1, f'Expected time (first) dimension to be equal to one. Found shape: {z0.shape[0]}.'
        z0 = z0[0]

        z0_mean, z0_log_var = z0[:, :self._latent_dim], z0[:, self._latent_dim:]
        return z0_mean, z0_log_var


class ODERNNVAE(nn.Module):
    """
    ODE-RNN-VAE
    """
    def __init__(
        self,
        observable_dim: int,
        hidden_dim: int,
        latent_dim: int,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None
    ):
        super().__init__()

        self._encoder = ODERNNVAEEncoder(
            observable_dim=observable_dim+1, # time is additional obs dimension
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            solver_name=solver_name,
            solver_params=solver_params
        )
        self._decoder = NODEDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=observable_dim,
            solver_name=solver_name,
            solver_params=solver_params
        )

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None, generate: bool = False) \
            -> Tuple[torch.Tensor, ...]:
        xt = torch.cat([x, t_obs], dim=-1)
        z0_mean, z0_log_var = self._encoder(xt)
        z0 = z0_mean if not generate else z0_mean + torch.randn_like(z0_mean) * torch.exp(0.5 * z0_log_var)
        x_hat = self._decoder(z0, t_unobs)
        return x_hat, z0_mean, z0_log_var


class LightningODERNNVAE(LightningModuleBase):
    """
    PytorchLightning wrapper for ODERNNVAE model
    """
    def __init__(
        self,
        observable_dim: int,
        hidden_dim: int,
        latent_dim: int,

        noise_std: float = 0.1,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None,

        train_config: Optional[LightningTrainConfig] = None
    ):
        super().__init__(train_config=train_config)
        self._model = ODERNNVAE(observable_dim, hidden_dim, latent_dim, solver_name=solver_name, solver_params=solver_params)
        self._loss_func = ELBO(noise_std)

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None, generate: bool = False) \
            -> Tuple[torch.Tensor, ...]:
        return self._model(x, t_obs, t_unobs, generate)

    def training_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, _ = batch
        bboxes_unobs_hat, z0_mean, z0_log_var = self.forward(bboxes_obs, ts_obs, ts_unobs)
        loss, kl_div_loss, likelihood_loss = self._loss_func(bboxes_unobs_hat, bboxes_unobs, z0_mean, z0_log_var)

        self._meter.push('training/loss', loss)
        self._meter.push('training/kl_div_loss', kl_div_loss)
        self._meter.push('training/likelihood_loss', likelihood_loss)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, _ = batch
        bboxes_unobs_hat, z0_mean, z0_log_var = self.forward(bboxes_obs, ts_obs, ts_unobs)
        loss, kl_div_loss, likelihood_loss = self._loss_func(bboxes_unobs_hat, bboxes_unobs, z0_mean, z0_log_var)

        self._meter.push('val/loss', loss)
        self._meter.push('val/kl_div_loss', kl_div_loss)
        self._meter.push('val/likelihood_loss', likelihood_loss)

        # Repeated for compatibility with ODEVAE model
        self._meter.push('val-forecast/loss', loss)
        self._meter.push('val-forecast/kl_div_loss', kl_div_loss)
        self._meter.push('val-forecast/likelihood_loss', likelihood_loss)

        return loss


def run_test():
    odernnvae = ODERNNVAE(
        observable_dim=7,
        hidden_dim=5,
        latent_dim=3
    )

    # Test ODERNNVAE
    xs = torch.randn(4, 3, 7)
    ts_obs = torch.randn(4, 3, 1)
    ts_unobs = torch.randn(2, 3, 1)
    expected_shapes = [(2, 3, 7), (3, 3), (3,3)]

    output = odernnvae(xs, ts_obs, ts_unobs)
    shapes = [v.shape for v in output]
    assert shapes == expected_shapes, f'Expected shapes {expected_shapes} but found {shapes}!'


if __name__ == '__main__':
    run_test()