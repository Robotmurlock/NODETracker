"""
Generative latent functions time-series model
"""
from typing import Tuple, Optional, Dict, Union

import torch
from torch import nn

from nodetracker.library import time_series
from nodetracker.node.core import ODEF, NeuralODE, ode_solver_factory
from nodetracker.node.losses.vae import ELBO
from nodetracker.node.utils import LightningModuleBase, LightningTrainConfig


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


class MLPODEFWithGlobalState(ODEF):
    """
    Multi layer perceptron ODEF. Includes N consecutive layers of:
    - Linear layer
    - LayerNorm
    - LeakyReLU
    """
    def __init__(self, dim: int, hidden_dim: int, n_layers: int = 2):
        assert dim % 2 == 0 and hidden_dim % 2 == 0, f'Dimensions must by divisable by 2. Got {dim} and {hidden_dim}.'

        super(ODEF, self).__init__()
        assert n_layers >= 1, f'Minimum number of layers is 1 but found {n_layers}'
        layers_args = [[hidden_dim, hidden_dim // 2] for _ in range(n_layers)]
        layers_args[0][0] = dim + 1  # setting input dimension (including time dimension)
        layers_args[-1][1] = dim // 2 # setting output dimension

        self._model = nn.ModuleList([self._create_mlp_layer(*args) for args in layers_args])

    @staticmethod
    def _create_mlp_layer(input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(0.1)
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        split_dim = -1
        size = z.shape[split_dim] // 2
        _, g = torch.split(z, size, dim=split_dim)

        zt = torch.cat([z, t], dim=-1)
        last_layer_index = len(self._model) - 1

        for layer_index, layer in enumerate(self._model):
            zt = layer(zt)
            if layer_index != last_layer_index:
                zt = torch.cat([zt, g], dim=-1)

        zero_mask = torch.zeros_like(g).to(g)
        zero_mask.requires_grad_(False)
        return torch.cat([zt, zero_mask], dim=-1)


class NODEDecoder(nn.Module):
    """
    NODE decoder performs extrapolation at latent space which can be then used to reconstruct/forecast time-series.
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,

        model_gaussian: bool = False,

        n_mlp_layers: int = 2,
        global_state: bool = False,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None
    ):
        super().__init__()

        if global_state:
            func = MLPODEFWithGlobalState(latent_dim, hidden_dim, n_layers=n_mlp_layers)
        else:
            func = MLPODEF(latent_dim, hidden_dim, n_layers=n_mlp_layers)

        solver = ode_solver_factory(solver_name, solver_params)

        self._ode = NeuralODE(func=func, solver=solver)
        self._latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self._lrelu = nn.LeakyReLU(0.1)

        output_dim = output_dim if not model_gaussian else 2 * output_dim  # mean + std in case of gaussian modeling
        self._hidden2output = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, z0: torch.Tensor, ts: torch.Tensor):
        _, batch_size, _ = ts.shape
        t0 = torch.zeros(1, batch_size, 1, dtype=torch.float32).to(ts)
        ts_all = torch.cat([t0, ts], dim=0)
        zs = self._ode(z0, ts_all, full_sequence=True)
        zs = zs[1:]
        hs = self._lrelu(self._latent2hidden(zs))
        xs = self._hidden2output(hs)
        return xs, zs


class ODEVAE(nn.Module):
    """
    ODEVAE - A generative latent function time-series model
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

        self._encoder = RNNEncoder(observable_dim, hidden_dim, latent_dim)
        self._decoder = NODEDecoder(latent_dim, hidden_dim, observable_dim,
                                    solver_name=solver_name, solver_params=solver_params)

    def forward(
        self,
        x: torch.Tensor,
        t_obs: torch.Tensor,
        t_unobs: Optional[torch.Tensor] = None,
        generate: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        n_obs = t_obs.shape[0]
        t_all = torch.cat([t_obs, t_unobs], dim=0) if t_unobs is not None else t_obs

        z0_mean, z0_log_var = self._encoder(x, t_obs)
        z0 = z0_mean if not generate else z0_mean + torch.randn_like(z0_mean) * torch.exp(0.5 * z0_log_var)
        x_hat_all, z_hat_all = self._decoder(z0, t_all)
        x_unobs_hat = x_hat_all[n_obs:, :, :]
        z_hat_hat = z_hat_all[n_obs:, :, :]

        return x_unobs_hat, x_hat_all, z0_mean, z0_log_var, z_hat_all, z_hat_hat


class LightningODEVAE(LightningModuleBase):
    """
    PytorchLightning wrapper for ODEVAE model
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
        self._model = ODEVAE(observable_dim, hidden_dim, latent_dim,
                             solver_name=solver_name, solver_params=solver_params)
        assert train_config.loss_name.lower() == 'elbo', 'ODEVAE only supports ELBO loss!'
        self._loss_func = ELBO(noise_std)

    def forward(
        self,
        x: torch.Tensor,
        t_obs: torch.Tensor,
        t_unobs: Optional[torch.Tensor] = None,
        generate: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        return self._model(x, t_obs, t_unobs, generate)

    def training_step(self, batch: Dict[str, Union[dict, torch.Tensor]], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_aug_unobs, ts_obs, ts_unobs, orig_bboxes_obs, orig_bboxes_unobs, bboxes_unobs, metadata = batch.values()
        bboxes_all = torch.cat([bboxes_obs, bboxes_unobs], dim=0)

        _, bboxes_hat_all, z0_mean, z0_log_var = self.forward(bboxes_obs, ts_obs, ts_unobs)
        loss, kl_div_loss, likelihood_loss = self._loss_func(bboxes_hat_all, bboxes_all, z0_mean, z0_log_var)

        self._meter.push('training/loss', loss)
        self._meter.push('training/kl_div_loss', kl_div_loss)
        self._meter.push('training/likelihood_loss', likelihood_loss)

        return loss

    def validation_step(self, batch: Dict[str, Union[dict, torch.Tensor]], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_aug_unobs, ts_obs, ts_unobs, orig_bboxes_obs, orig_bboxes_unobs, bboxes_unobs, metadata = batch.values()
        bboxes_all = torch.cat([bboxes_obs, bboxes_unobs], dim=0)

        bboxes_unobs_hat, bboxes_all_hat, z0_mean, z0_log_var = self.forward(bboxes_obs, ts_obs, ts_unobs)
        all_loss, all_kl_div_loss, all_likelihood_loss = \
            self._loss_func(bboxes_all_hat, bboxes_all, z0_mean, z0_log_var)
        unobs_loss, unobs_kl_div_loss, unobs_likelihood_loss = \
            self._loss_func(bboxes_unobs_hat, bboxes_unobs, z0_mean, z0_log_var)

        self._meter.push('val/loss', all_loss)
        self._meter.push('val/kl_div_loss', all_kl_div_loss)
        self._meter.push('val/likelihood_loss', all_likelihood_loss)

        self._meter.push('val-forecast/loss', unobs_loss)
        self._meter.push('val-forecast/kl_div_loss', unobs_kl_div_loss)
        self._meter.push('val-forecast/likelihood_loss', unobs_likelihood_loss)

        return all_loss


def run_test():
    model = ODEVAE(4, 3, 2)
    x_obs = torch.randn(3, 1, 4)
    t_obs = torch.randn(3, 1, 1)
    t_unobs = torch.randn(2, 1, 1)

    print('Output:', [x.shape for x in model(x_obs, t_obs, t_unobs)])


if __name__ == '__main__':
    run_test()
