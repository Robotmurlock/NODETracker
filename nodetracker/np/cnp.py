"""
Implementation of Conditional Neural Processes for time series
"""
from typing import Optional, Union

import torch
from torch import nn

from nodetracker.datasets.transforms import InvertibleTransform, InvertibleTransformWithVariance
from nodetracker.node.odernn.utils import LightningGaussianModel, run_simple_lightning_guassian_model_test
from nodetracker.node.utils import LightningTrainConfig
from nodetracker.np.core.cnp import CNP
from nodetracker.np.utils import to_scaled_relative_ts


class BaselineCNP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        n_classes: Optional[int] = None,

        hidden_dim: int = 8,
        n_input2hidden_layers: int = 2,
        n_target2hidden_layers: int = 2,
        n_enc_layers: int = 2,

        n_head_layers: int = 2,
        t_scale: float = 5.0
    ):
        super().__init__()

        self._cnp = CNP(
            input_dim=input_dim,
            target_dim=target_dim,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            n_input2hidden_layers=n_input2hidden_layers,
            n_target2hidden_layers=n_target2hidden_layers,
            n_enc_layers=n_enc_layers,
            n_head_layers=n_head_layers
        )

        self._t_scale = t_scale

    def forward(self, x_obs: torch.Tensor, ts_obs: torch.Tensor, ts_unobs: torch.Tensor, metadata: dict) -> torch.Tensor:
        with torch.no_grad():
            ts_obs, ts_unobs = to_scaled_relative_ts(ts_obs, ts_unobs, self._t_scale)

        _ = metadata  # Ignored
        return self._cnp(ts_obs, x_obs, ts_unobs)


class LightningBaselineCNP(LightningGaussianModel):
    """
    Trainer wrapper for CNP.
    """
    def __init__(
        self,
        observable_dim: int,
        hidden_dim: int,

        n_input2hidden_layers: int = 2,
        n_target2hidden_layers: int = 2,
        n_enc_layers: int = 2,

        n_head_layers: int = 2,

        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None,

        train_config: Optional[LightningTrainConfig] = None,
        log_epoch_metrics: bool = True
    ):
        """
        Args:
            observable_dim: Trajectory point dimension
            hidden_dim: Hidden trajectory dimension
        """
        model = BaselineCNP(
            input_dim=1,
            target_dim=observable_dim,
            hidden_dim=hidden_dim,
            n_input2hidden_layers=n_input2hidden_layers,
            n_target2hidden_layers=n_target2hidden_layers,
            n_enc_layers=n_enc_layers,
            n_head_layers=n_head_layers
        )
        super().__init__(
            train_config=train_config,
            model=model,
            model_gaussian=True,
            transform_func=transform_func,
            log_epoch_metrics=log_epoch_metrics
        )


def main() -> None:
    run_simple_lightning_guassian_model_test(
        model_class=LightningBaselineCNP,
        params={
            'observable_dim': 7,
            'hidden_dim': 3,
        },
        model_gaussian_only=True
    )


if __name__ == '__main__':
    main()
