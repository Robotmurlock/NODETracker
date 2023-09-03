"""
Custom ODE-RNN-VAE implementation
https://arxiv.org/pdf/1907.03907.pdf
"""
from typing import Tuple, Optional, List, Dict, Union

import torch
from torch import nn

from nodetracker.library.building_blocks import MLP
from nodetracker.node.core.odevae import MLPODEF, NODEDecoder, ELBO
from nodetracker.node.core.original import NeuralODE
from nodetracker.node.core.solver.factory import ode_solver_factory
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

    # noinspection DuplicatedCode
    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reverse input time sequence (running RNN backward in time)
        xs = torch.flip(xs, dims=[0])
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
    ODE-RNN-VAE implementation
    """
    def __init__(
        self,
        observable_dim: int,
        hidden_dim: int,
        latent_dim: int,
        stochastic: bool = True,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None
    ):
        """
        Args:
            observable_dim: Trajectory point dimension
            hidden_dim: Hidden dimension
            latent_dim: Latent dimension
            stochastic: Use re-parameterization trick (default: True)
            solver_name: ODE solver name
            solver_params: ODE solver params
        """
        super().__init__()
        self._stochastic = stochastic

        self._encoder = ODERNNVAEEncoder(
            observable_dim=observable_dim+1,  # time is additional obs dimension
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

    def forward(
        self,
        x: torch.Tensor,
        t_obs: torch.Tensor,
        t_unobs: Optional[torch.Tensor] = None,
        n_samples: int = 1
    ) -> Tuple[torch.Tensor, ...]:
        xt = torch.cat([x, t_obs], dim=-1)
        z0_mean, z0_log_var = self._encoder(xt)

        z0 = z0_mean  # shape: (batch_size, latent_dim)
        if self._stochastic:
            assert n_samples >= 1, f'Number of samples must be at least 1! Got {n_samples}.'
            # Generation:
            # std = sqrt(var)
            # var = exp(log_var)
            # => std = sqrt(exp(log_var)) = exp(log_var/2) = exp(0.5 * log_var)
            # noise := r * std, where r is from N(0, 1) - normal distribution

            z0_samples: List[torch.Tensor] = []
            for _ in range(n_samples):
                noise = torch.randn_like(z0_mean) * torch.exp(0.5 * z0_log_var)
                z0_samples.append(z0 + noise)

            z0 = torch.cat(z0_samples, dim=0)  # shape: (n_samples*batch_size, latent_dim)
            t_unobs = t_unobs.repeat(1, n_samples, 1)

        x_hat, z_hat = self._decoder(z0, t_unobs)
        return x_hat, z_hat, z0_mean, z0_log_var

    def predict_monte_carlo(
        self,
        x: torch.Tensor,
        t_obs: torch.Tensor,
        t_unobs: Optional[torch.Tensor] = None,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Estimate model mean and std using Monte Carlo sampling from latent space.
        For each trajectory in batch:
            - Sample `n_samples` from estimated VAE posterior distribution
            - Propagate all samples through decoder

        Args:
            x: Features (trajectory)
            t_obs: Observed time points
            t_unobs: Unobserved time points
            n_samples: Number of samples to estimate mean and variance

        Returns:
            Grouped trajectory samples
        """
        unobs_time_len, batch_size, _ = t_unobs.shape
        x_hat, *_ = self.forward(x, t_obs, t_unobs, n_samples=n_samples)
        return x_hat.view(unobs_time_len, batch_size, n_samples, -1)


class LightningODERNNVAE(LightningModuleBase):
    """
    PytorchLightning wrapper for ODERNNVAE model
    """
    def __init__(
        self,
        observable_dim: int,
        hidden_dim: int,
        latent_dim: int,
        stochastic: bool = True,

        noise_std: float = 0.1,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None,

        train_config: Optional[LightningTrainConfig] = None
    ):
        super().__init__(train_config=train_config)
        self._model = ODERNNVAE(
            observable_dim=observable_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            stochastic=stochastic,
            solver_name=solver_name,
            solver_params=solver_params
        )
        assert train_config.loss_name.lower() == 'elbo', 'ODERNNVAE only supports ELBO loss!'
        self._loss_func = ELBO(noise_std)

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, ...]:
        return self._model(x, t_obs, t_unobs)

    def training_step(self, batch: Dict[str, Union[dict, torch.Tensor]], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_aug_unobs, ts_obs, ts_unobs, orig_bboxes_obs, orig_bboxes_unobs, bboxes_unobs, metadata = batch.values()

        bboxes_obs_hat, _, z0_mean, z0_log_var = self.forward(bboxes_obs, ts_obs, ts_obs)
        loss, kl_div_loss, likelihood_loss = self._loss_func(bboxes_obs_hat, bboxes_obs, z0_mean, z0_log_var)

        self._meter.push('training/loss', loss)
        self._meter.push('training/kl_div_loss', kl_div_loss)
        self._meter.push('training/likelihood_loss', likelihood_loss)

        return loss

    def validation_step(self, batch: Dict[str, Union[dict, torch.Tensor]], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_aug_unobs, ts_obs, ts_unobs, orig_bboxes_obs, orig_bboxes_unobs, bboxes_unobs, metadata = batch.values()

        bboxes_obs_hat, _, z0_mean, z0_log_var = self.forward(bboxes_obs, ts_obs, ts_obs)
        loss, kl_div_loss, likelihood_loss = self._loss_func(bboxes_obs_hat, bboxes_obs, z0_mean, z0_log_var)

        self._meter.push('val/loss', loss)
        self._meter.push('val/kl_div_loss', kl_div_loss)
        self._meter.push('val/likelihood_loss', likelihood_loss)

        ts_all = torch.cat([ts_obs, ts_unobs])
        bboxes_all = torch.cat([bboxes_obs, bboxes_unobs])
        bboxes_all_hat, _, z0_mean, z0_log_var = self.forward(bboxes_obs, ts_obs, ts_all)
        loss, kl_div_loss, likelihood_loss = self._loss_func(bboxes_all_hat, bboxes_all, z0_mean, z0_log_var)
        self._meter.push('val-forecast/loss', loss)
        self._meter.push('val-forecast/kl_div_loss', kl_div_loss)
        self._meter.push('val-forecast/likelihood_loss', likelihood_loss)

        return loss

    def predict_monte_carlo(
        self,
        x: torch.Tensor,
        t_obs: torch.Tensor,
        t_unobs: Optional[torch.Tensor] = None,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Estimated mean and std for each trajectory (check model docstring for more info)

        Args:
            x: Features (trajectory)
            t_obs: Observed time points
            t_unobs: Unobserved time points
            n_samples: Number of samples to estimate mean and variance

        Returns:
            Prediction mean and std
        """
        return self._model.predict_monte_carlo(x, t_obs, t_unobs, n_samples=n_samples)


def run_test():
    odernnvae = ODERNNVAE(
        observable_dim=7,
        hidden_dim=5,
        latent_dim=3
    )

    # Test ODERNNVAE prediction
    xs = torch.randn(4, 3, 7)
    ts_obs = torch.randn(4, 3, 1)
    ts_unobs = torch.randn(2, 3, 1)
    expected_shapes = [(2, 3, 7), (2, 3, 3), (3, 3), (3, 3)]

    output = odernnvae(xs, ts_obs, ts_unobs)
    shapes = [v.shape for v in output]
    assert shapes == expected_shapes, f'Expected shapes {expected_shapes} but found {shapes}!'

    # Test ODERNNVAE generate
    expected_shapes = [(2, 6, 7), (2, 6, 3), (3, 3), (3, 3)]

    output = odernnvae(xs, ts_obs, ts_unobs, generate=True, n_samples=2)
    shapes = [v.shape for v in output]
    assert shapes == expected_shapes, f'Expected shapes {expected_shapes} but found {shapes}!'

    # Test ODERNNVAE estimate - Monte Carlo
    expected_shapes = [(2, 3, 7), (2, 3, 7)]

    output = odernnvae.predict_monte_carlo(xs, ts_obs, ts_unobs, n_samples=10)
    shapes = [v.shape for v in output]
    assert shapes == expected_shapes, f'Expected shapes {expected_shapes} but found {shapes}!'


if __name__ == '__main__':
    run_test()
