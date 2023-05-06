"""
Custom model: RNN-ODE
Like ODE-RNN but without ODE-RNN in encoder
"""
from typing import Optional, Tuple, Union

import torch
from torch import nn

from nodetracker.datasets.transforms import InvertibleTransform, InvertibleTransformWithVariance
from nodetracker.library.building_blocks.mlp import MLP
from nodetracker.library.building_blocks.resnet import ResnetMLPBlock
from nodetracker.node.core.odevae import NODEDecoder
from nodetracker.node.odernn.utils import LightningGaussianModel, run_simple_lightning_guassian_model_test
from nodetracker.node.utils import LightningTrainConfig


class ResNetEncoder(nn.Module):
    """
    ResNet encoder for MLP-ODE model
    """
    def __init__(
        self,
        observable_steps: int,
        observable_dim: int,
        hidden_dim: int,

        n_stem_layers: int = 1,
        n_resnet_layers: int = 4,
        n_resnet_bottleneck_layers: int = 1
    ):
        super().__init__()
        self._flatten = nn.Flatten()
        self._stem = MLP(
            input_dim=observable_steps*(observable_dim+1),  # time dimension
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_layers=n_stem_layers
        )
        self._backbone = ResnetMLPBlock(
            dim=hidden_dim,
            n_layers=n_resnet_layers,
            n_bottleneck_layers=n_resnet_bottleneck_layers
        )

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None) -> torch.Tensor:
        _ = t_unobs  # Unused

        # Add relative time points to input data
        xt = torch.cat([x, t_obs], dim=-1)

        xt = torch.permute(xt, [1, 0, 2])  # (time, batch, bbox) -> (batch, time, bbox)
        xt = self._flatten(xt)  # (batch, time, bbox) -> (batch, time*bbox)
        output = self._stem(xt)  # (batch, steps*bbox)
        output = self._backbone(output)
        return output


class MLPODE(nn.Module):
    """
    MLP-ODE
    """
    def __init__(
        self,
        observable_steps: int,
        observable_dim: int,
        hidden_dim: int,

        model_gaussian: bool = False,

        n_stem_layers: int = 1,
        n_resnet_layers: int = 4,
        n_resnet_bottleneck_layers: int = 1,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None
    ):
        super().__init__()

        self._encoder = ResNetEncoder(
            observable_steps=observable_steps,
            observable_dim=observable_dim,
            hidden_dim=hidden_dim,
            n_stem_layers=n_stem_layers,
            n_resnet_layers=n_resnet_layers,
            n_resnet_bottleneck_layers=n_resnet_bottleneck_layers
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
        x_hat, z_hat = self._decoder(z0, t_unobs)
        return x_hat, z_hat


class LightningMLPODE(LightningGaussianModel):
    """
    PytorchLightning wrapper for MLPODE model.
    This model is similar to ODERNN but does not use ODE solver in encoder.
    """
    def __init__(
        self,
        observable_steps: int,
        observable_dim: int,
        hidden_dim: int,

        model_gaussian: bool = False,
        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None,

        n_stem_layers: int = 1,
        n_resnet_layers: int = 4,
        n_resnet_bottleneck_layers: int = 1,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None,

        train_config: Optional[LightningTrainConfig] = None
    ):
        model = MLPODE(
            observable_steps=observable_steps,
            observable_dim=observable_dim,
            hidden_dim=hidden_dim,
            n_stem_layers=n_stem_layers,
            n_resnet_layers=n_resnet_layers,
            n_resnet_bottleneck_layers=n_resnet_bottleneck_layers,
            solver_name=solver_name,
            solver_params=solver_params,
            model_gaussian=model_gaussian
        )
        super().__init__(
            train_config=train_config,
            model=model,
            model_gaussian=model_gaussian,
            transform_func=transform_func
        )

        self._is_modeling_gaussian = model_gaussian

if __name__ == '__main__':
    run_simple_lightning_guassian_model_test(
        model_class=LightningMLPODE,
        params={
            'observable_steps': 4,
            'observable_dim': 7,
            'hidden_dim': 4
        }
    )
