"""
ODE-RNN implementation
https://arxiv.org/pdf/1907.03907.pdf
"""
from typing import Optional, Tuple

import torch
from torch import nn

from nodetracker.node.building_blocks import MLP
from nodetracker.node.core.odevae import MLPODEF, NODEDecoder
from nodetracker.node.core.original import NeuralODE
from nodetracker.node.core.solver import ode_solver_factory
from nodetracker.node.utils import LightningTrainConfig, LightningModuleForecaster


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

        model_gaussian: bool = False,

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
            solver_params=solver_params,
            model_gaussian=model_gaussian
        )

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        xt = torch.cat([x, t_obs], dim=-1)
        z0 = self._encoder(xt)
        x_hat, z_hat = self._decoder(z0, t_unobs)
        return x_hat, z_hat


class LightningODERNN(LightningModuleForecaster):
    """
    PytorchLightning wrapper for ODERNN model
    """
    def __init__(
        self,
        observable_dim: int,
        hidden_dim: int,

        model_gaussian: bool = False,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None,

        train_config: Optional[LightningTrainConfig] = None
    ):
        model = ODERNN(
            observable_dim=observable_dim,
            hidden_dim=hidden_dim,
            model_gaussian=model_gaussian,
            solver_name=solver_name,
            solver_params=solver_params
        )
        loss_func = nn.GaussianNLLLoss() if model_gaussian else nn.MSELoss()
        super().__init__(train_config=train_config, model=model, loss_func=loss_func, model_gaussian=model_gaussian)

    @staticmethod
    def extract_mean_and_std(bboxes_unobs_hat: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function for Gaussian model postprocess

        Args:
            bboxes_unobs_hat: Prediction

        Returns:
            bboxes_hat_mean, bboxes_hat_std
        """
        bboxes_unobs_hat = bboxes_unobs_hat.view(*bboxes_unobs_hat.shape[:-1], -1, 2)
        bboxes_unobs_hat_mean = bboxes_unobs_hat[..., 0]
        bboxes_unobs_hat_log_var = bboxes_unobs_hat[..., 1]
        bboxes_unobs_hat_var = torch.exp(bboxes_unobs_hat_log_var)
        bboxes_unobs_hat_std = torch.sqrt(bboxes_unobs_hat_var)

        return bboxes_unobs_hat_mean, bboxes_unobs_hat_std

    def inference(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        x_hat, *other = self._model(x, t_obs, t_unobs)

        if self._model_gaussian:
            x_hat_mean, x_hat_std = self.extract_mean_and_std(x_hat)
            return x_hat_mean, x_hat_std, *other

        return x_hat, *other


def run_test():
    # Test ODERNNVAE
    xs = torch.randn(4, 3, 7)
    ts_obs = torch.randn(4, 3, 1)
    ts_unobs = torch.randn(2, 3, 1)
    expected_shape = (2, 3, 7)

    # Test standard ODERNN
    odernn = LightningODERNN(
        observable_dim=7,
        hidden_dim=5
    )

    output, _ = odernn(xs, ts_obs, ts_unobs)
    assert output.shape == expected_shape, f'Expected shape {expected_shape} but found {output.shape}!'

    # Test ODERNN with gaussian modeling
    odernn = LightningODERNN(
        observable_dim=7,
        hidden_dim=5,
        model_gaussian=True
    )

    expected_shape = (2, 3, 14)
    output, _ = odernn(xs, ts_obs, ts_unobs)
    assert output.shape == expected_shape, f'Expected shape {expected_shape} but found {output.shape}!'

    expected_shapes = [(2, 3, 7), (2, 3, 7)]
    output = odernn.extract_mean_and_std(output)
    output_shapes = [o.shape for o in output]
    assert output_shapes == expected_shapes, f'Expected shape {expected_shape} but found {output_shapes}!'


if __name__ == '__main__':
    run_test()
