"""
Custom model: RNN-ODE
Like ODE-RNN but without ODE in encoder
"""
from typing import Optional, Tuple, Union, List

import torch
from torch import nn

from nodetracker.datasets.transforms import InvertibleTransform, InvertibleTransformWithVariance
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
        z0 = z0[-1]  # Removing temporal dim
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
        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None,

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
            model_gaussian=model_gaussian,
            transform_func=transform_func
        )


class ComposeRNNODE(nn.Module):
    def __init__(self, n_categories: int, *args, **kwargs):
        super().__init__()
        self._models = nn.ModuleDict({str(i): RNNODE(*args, **kwargs) for i in range(n_categories)})

    def forward(self, x_obs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # BBox encoder
        bbox_obs = x_obs[..., :-1]

        # Transform category index to embedding
        if len(x_obs.shape) == 2:
            x_category = str(int(x_obs[0, -1].item()))
            model = self._models[x_category]
            return model(bbox_obs, t_obs, t_unobs)
        elif len(x_obs.shape) == 3:
            x_categories = x_obs[0, :, -1]
            outputs: List[torch.Tensor] = []
            for batch_index in range(x_categories.shape[0]):
                x_category = str(int(x_categories[batch_index].item()))
                model = self._models[x_category]
                output = model(
                    bbox_obs[:, batch_index:batch_index+1, :],
                    t_obs[:, batch_index:batch_index+1, :],
                    t_unobs[:, batch_index:batch_index+1, :]
                )
                outputs.append(output)

            return tuple(torch.cat(o, dim=1) for o in zip(*outputs))
        else:
            raise AssertionError(f'Unexpected shape {x_obs.shape}.')


class LightningComposeRNNODE(LightningGaussianModel):
    """
    Compose of RNNODE models. Requires added labels.
    """
    def __init__(
        self,
        model_gaussian: bool = False,
        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None,
        train_config: Optional[LightningTrainConfig] = None,
        *args,
        **kwargs
    ):
        model = ComposeRNNODE(model_gaussian=model_gaussian, *args, **kwargs)

        super().__init__(
            train_config=train_config,
            model=model,
            model_gaussian=model_gaussian,
            transform_func=transform_func
        )


if __name__ == '__main__':
    run_simple_lightning_guassian_model_test(
        model_class=LightningRNNODE,
        params={
            'observable_dim': 7,
            'hidden_dim': 4,

            'n_encoder_rnn_layers': 2
        }
    )
