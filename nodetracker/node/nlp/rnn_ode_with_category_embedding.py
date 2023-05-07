"""
Custom modified model: RNN-ODE
Model is modified to support bbox category embedding
"""
from typing import Optional, Tuple, Union

import torch
from torch import nn

from nodetracker.datasets.transforms import InvertibleTransform, InvertibleTransformWithVariance
from nodetracker.node.core.odevae import NODEDecoder
from nodetracker.node.odernn.utils import LightningGaussianModel
from nodetracker.node.utils import LightningTrainConfig
from nodetracker.standard.rnn.seq_to_seq import RNNEncoder


class CategoryRNNEncoder(nn.Module):
    """
    Time-series RNN encoder. Can work with time-series with variable lengths and possible missing values.
    Supports Category input
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_embeddings: int,
        embedding_dim: int,

        rnn_n_layers: int = 1,
    ):
        """
        Args:
            input_dim: Trajectory point dimension
            hidden_dim: Hidden trajectory dimension
            latent_dim: "latent" (hidden-2) trajectory dimension
            rnn_n_layers: Number of stacked RNN (GRU) layers
        """
        super().__init__()
        self._bbox_encoder = RNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            rnn_n_layers=rnn_n_layers
        )
        self._category_encoder = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        )

    def forward(self, x_obs: torch.Tensor, t_obs: torch.Tensor) -> torch.Tensor:
        # BBox encoder
        bbox_obs = x_obs[..., :-1]
        z = self._bbox_encoder(bbox_obs, t_obs)

        # Transform category index to embedding
        x_category = x_obs[0, -1] if len(x_obs.shape) == 2 else x_obs[0, :, -1]
        x_category = x_category.unsqueeze(0).long()
        z_embedding = self._category_encoder(x_category)
        z = torch.concat([z, z_embedding], dim=-1)
        return z


class CategoryRNNODE(nn.Module):
    """
    Category RNN-ODE
    """
    def __init__(
        self,
        observable_dim: int,
        hidden_dim: int,

        num_embeddings: int,
        embedding_dim: int,

        model_gaussian: bool = False,

        n_encoder_rnn_layers: int = 1,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None
    ):
        super().__init__()

        self._encoder = CategoryRNNEncoder(
            input_dim=observable_dim,  # time is additional obs dimension
            hidden_dim=hidden_dim,
            latent_dim=hidden_dim,

            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,

            rnn_n_layers=n_encoder_rnn_layers
        )
        self._decoder = NODEDecoder(
            latent_dim=hidden_dim + embedding_dim,
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


class LightningCategoryRNNODE(LightningGaussianModel):
    """
    PytorchLightning wrapper for RNNODE model.
    This model is similar to ODERNN but does not use ODE solver in encoder.
    """
    def __init__(
        self,
        observable_dim: int,
        hidden_dim: int,

        num_embeddings: int,
        embedding_dim: int,

        model_gaussian: bool = False,
        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None,

        n_encoder_rnn_layers: int = 1,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None,

        train_config: Optional[LightningTrainConfig] = None
    ):
        model = CategoryRNNODE(
            observable_dim=observable_dim,
            hidden_dim=hidden_dim,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,

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


def run_test() -> None:
    # Test Model
    xs = torch.randn(4, 3, 7)
    category = torch.ones(*xs.shape[:-1], 1)
    xs = torch.concat([xs, category], dim=-1)

    ts_obs = torch.randn(4, 3, 1)
    ts_unobs = torch.randn(2, 3, 1)

    # Test standard (no gaussian)
    model = LightningCategoryRNNODE(
        observable_dim=7,
        hidden_dim=8,
        num_embeddings=5,
        embedding_dim=3,
    )

    output = model(xs, ts_obs, ts_unobs)
    print([o.shape for o in output])


if __name__ == '__main__':
    run_test()
