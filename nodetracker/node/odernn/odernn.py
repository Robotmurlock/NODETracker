"""
ODE-RNN implementation
https://arxiv.org/pdf/1907.03907.pdf
"""
from typing import Optional, Tuple, Union

import torch
from torch import nn

from nodetracker.datasets.transforms import InvertibleTransform, InvertibleTransformWithVariance
from nodetracker.library.building_blocks import MLP
from nodetracker.node.core.odevae import MLPODEF, NODEDecoder
from nodetracker.node.core.original import NeuralODE
from nodetracker.node.core.solver.factory import ode_solver_factory
from nodetracker.node.odernn.utils import LightningGaussianModel, run_simple_lightning_guassian_model_test
from nodetracker.node.utils import LightningTrainConfig


class ODERNNEncoder(nn.Module):
    """
    ODE-RNN Encoder. Encoder uses combination of NODE and RNN (NODE -> RNN).
    """

    def __init__(
            self,
            observable_dim: int,
            hidden_dim: int,
            n_node_mlp_layers: int = 2,

            n_rnn_layers: int = 1,

            solver_name: Optional[str] = None,
            solver_params: Optional[dict] = None
    ):
        """
        Args:
            observable_dim: Data dimension
            hidden_dim: Hidden dimension (additional dimension between observable and hidden)
            n_node_mlp_layers: Number NODE MLP layers

            n_rnn_layers: Number of RNN layers

            solver_name: ODE solver name
            solver_params: ODE solver params
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
            num_layers=n_rnn_layers,
            batch_first=False
        )
        self._n_rnn_layers = n_rnn_layers

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        time_len, batch_size, _ = xs.shape

        hs = self._obs2hidden(xs)
        prev_h0 = torch.zeros(batch_size, self._hidden_dim).to(hs)

        h0 = None
        for i in range(time_len):
            ts = torch.tensor([0, 1], dtype=torch.float32).view(2, 1, 1).expand(2, batch_size, 1).to(hs)

            prev_h0 = self._node(prev_h0, ts).unsqueeze(0)
            h0 = hs[i:i+1]
            h0, prev_h0 = self._rnn(h0, prev_h0)
            prev_h0 = prev_h0[0]

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
        output_dim: int,

        model_gaussian: bool = False,

        n_encoder_rnn_layers: int = 1,
        n_decoder_mlp_layers: int = 2,
        decoder_global_state: bool = False,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None,
        encoder_solver_name: Optional[str] = None,
        encoder_solver_params: Optional[dict] = None,
        decoder_solver_name: Optional[str] = None,
        decoder_solver_params: Optional[dict] = None
    ):
        super().__init__()
        # Back-compatibility
        encoder_solver_name = solver_name if encoder_solver_name is None else encoder_solver_name
        encoder_solver_params = solver_params if encoder_solver_params is None else encoder_solver_params
        decoder_solver_name = solver_name if decoder_solver_name is None else decoder_solver_name
        decoder_solver_params = solver_params if decoder_solver_params is None else decoder_solver_params

        self._encoder = ODERNNEncoder(
            observable_dim=observable_dim + 1,  # time is additional obs dimension
            hidden_dim=hidden_dim,
            solver_name=encoder_solver_name,
            solver_params=encoder_solver_params,
            n_rnn_layers=n_encoder_rnn_layers
        )
        self._decoder = NODEDecoder(
            latent_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_mlp_layers=n_decoder_mlp_layers,
            solver_name=decoder_solver_name,
            solver_params=decoder_solver_params,
            model_gaussian=model_gaussian,
            global_state=decoder_global_state
        )

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None, metadata: Optional[dict] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        _ = metadata  # ignored

        xt = torch.cat([x, t_obs], dim=-1)
        z0 = self._encoder(xt)

        # Decoder has time points relative to last observed time points
        # This is very important for long-term forecasting
        t_last = t_obs[-1, :, :]
        t_unobs = t_unobs - t_last.expand_as(t_unobs)
        x_hat, z_hat = self._decoder(z0, t_unobs)
        return x_hat, z_hat


class LightningODERNN(LightningGaussianModel):
    """
    PytorchLightning wrapper for ODERNN model
    """
    def __init__(
        self,
        observable_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,

        model_gaussian: bool = False,
        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None,

        n_encoder_rnn_layers: int = 1,
        n_decoder_mlp_layers: int = 2,
        decoder_global_state: bool = False,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None,
        encoder_solver_name: Optional[str] = None,
        encoder_solver_params: Optional[dict] = None,
        decoder_solver_name: Optional[str] = None,
        decoder_solver_params: Optional[dict] = None,

        train_config: Optional[LightningTrainConfig] = None,
        log_epoch_metrics: bool = True
    ):
        if output_dim is None:
            output_dim = observable_dim

        model = ODERNN(
            observable_dim=observable_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            model_gaussian=model_gaussian,
            solver_name=solver_name,
            solver_params=solver_params,
            encoder_solver_name=encoder_solver_name,
            encoder_solver_params=encoder_solver_params,
            decoder_solver_name=decoder_solver_name,
            decoder_solver_params=decoder_solver_params,
            n_encoder_rnn_layers=n_encoder_rnn_layers,
            n_decoder_mlp_layers=n_decoder_mlp_layers,
            decoder_global_state=decoder_global_state
        )
        super().__init__(
            train_config=train_config,
            model=model,
            model_gaussian=model_gaussian,
            transform_func=transform_func,
            log_epoch_metrics=log_epoch_metrics
        )


if __name__ == '__main__':
    configurations = [
        {
            'observable_dim': 7,
            'hidden_dim': 4
        },
        {
            'observable_dim': 7,
            'hidden_dim': 4,
            'decoder_global_state': True,
            'decoder_solver_name': 'euler_global',
            'decoder_solver_params': {
                'split_dim': -1,
                'max_step_size': 0.25
            }
        }
    ]

    for config in configurations:
        run_simple_lightning_guassian_model_test(
            model_class=LightningODERNN,
            params=config
        )
