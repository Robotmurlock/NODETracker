"""
ODE-RNN implementation
https://arxiv.org/pdf/1907.03907.pdf
"""
from typing import Tuple, Optional

import torch
from torch import nn

from nodetracker.node.building_blocks import MLP
from nodetracker.node.core.original import NeuralODE
from nodetracker.node.core.solver import ode_solver_factory
from nodetracker.node.generative_latent_time_series_model import MLPODEF, NODEDecoder
from nodetracker.node.utils import LightningTrainConfig, LightningModuleBase


class ODERNNEncoder(nn.Module):
    """
    ODE-RNN Encoder. Encoder uses combination of NODE and RNN (NODE -> RNN).
    """

    def __init__(
            self,
            observable_dim: int,
            hidden_dim: int,
            n_node_mlp_layers: int = 2,

            solver_name: Optional[str] = None,
            solver_params: Optional[dict] = None
    ):
        """
        Args:
            observable_dim: Data dimension
            hidden_dim: Hidden dimension (additional dimension between observable and hidden)
            n_node_mlp_layers: Number NODE MLP layers
        """
        super().__init__()
        assert n_node_mlp_layers >= 1, f'Minimum number of NODE MLP layers is 1 but found {n_node_mlp_layers}.'

        self._hidden_dim = hidden_dim
        self._obs2hidden = MLP(input_dim=observable_dim, output_dim=hidden_dim)

        odef_func = MLPODEF(
            dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=n_node_mlp_layers
        )

        solver = ode_solver_factory(name=solver_name, params=solver_params)
        self._node = NeuralODE(func=odef_func, solver=solver)
        self._rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=False
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        time_len, batch_size, _ = xs.shape

        hs = self._obs2hidden(xs)
        prev_h0 = torch.zeros(1, batch_size, self._hidden_dim).to(hs)

        h0 = None
        for i in range(time_len):
            ts = torch.tensor([0, 1], dtype=torch.float32).view(2, 1, 1).expand(2, batch_size, 1).to(hs)
            h0 = hs[i]

            h0 = self._node(h0, ts).unsqueeze(0)
            h0, prev_h0 = self._rnn(h0, prev_h0)

        assert h0 is not None, 'Zero ODERNN iterations performed!'
        assert h0.shape[0] == 1, f'Expected time (first) dimension to be equal to one. Found shape: {h0.shape[0]}.'
        h0 = h0[0]

        return h0


class ODERNN(nn.Module):
    """
    ODE-RNN
    """

    def __init__(
            self,
            observable_dim: int,
            hidden_dim: int,

            solver_name: Optional[str] = None,
            solver_params: Optional[dict] = None
    ):
        super().__init__()

        self._encoder = ODERNNEncoder(
            observable_dim=observable_dim + 1,  # time is additional obs dimension
            hidden_dim=hidden_dim,
            solver_name=solver_name,
            solver_params=solver_params
        )
        self._decoder = NODEDecoder(
            latent_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=observable_dim,
            solver_name=solver_name,
            solver_params=solver_params
        )

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None) -> torch.Tensor:
        xt = torch.cat([x, t_obs], dim=-1)
        z0 = self._encoder(xt)
        x_hat = self._decoder(z0, t_unobs)
        return x_hat


class LightningODERNN(LightningModuleBase):
    """
    PytorchLightning wrapper for ODERNN model
    """

    def __init__(
            self,
            observable_dim: int,
            hidden_dim: int,

            solver_name: Optional[str] = None,
            solver_params: Optional[dict] = None,

            train_config: Optional[LightningTrainConfig] = None
    ):
        super().__init__(train_config=train_config)
        self._model = ODERNN(observable_dim, hidden_dim, solver_name=solver_name, solver_params=solver_params)
        self._loss_func = nn.MSELoss()

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, ...]:
        return self._model(x, t_obs, t_unobs)

    def training_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, _ = batch
        bboxes_unobs_hat = self.forward(bboxes_obs, ts_obs, ts_unobs)
        loss = self._loss_func(bboxes_unobs_hat, bboxes_unobs)

        self._meter.push('training/loss', loss)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, _ = batch
        bboxes_unobs_hat = self.forward(bboxes_obs, ts_obs, ts_unobs)
        loss, kl_div_loss, likelihood_loss = self._loss_func(bboxes_unobs_hat, bboxes_unobs)

        self._meter.push('val/loss', loss)

        return loss


def run_test():
    # Test ODERNNVAE
    xs = torch.randn(4, 3, 7)
    ts_obs = torch.randn(4, 3, 1)
    ts_unobs = torch.randn(2, 3, 1)
    expected_shape = (2, 3, 7)

    odernn = LightningODERNN(
        observable_dim=7,
        hidden_dim=5
    )

    output = odernn(xs, ts_obs, ts_unobs)
    assert output.shape == expected_shape, f'Expected shape {expected_shape} but found {output.shape}!'


if __name__ == '__main__':
    run_test()