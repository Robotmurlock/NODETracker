"""
Implementation of simple MLP forecaster model.
"""
from typing import Optional, Union

import torch
from torch import nn

from nodetracker.library.building_blocks import MLP
from nodetracker.datasets.transforms import InvertibleTransform, InvertibleTransformWithVariance
from nodetracker.node.utils import LightningTrainConfig, LightningModuleForecaster


class MLPForecaster(nn.Module):
    """
    Simple MLP model.
    """
    def __init__(
        self,
        observable_steps: int,
        observable_dim: int,
        hidden_dim: int,
        forecast_steps: int,
        n_layers: int = 1
    ):
        """
        Args:
            observable_steps: Number of observable steps (input sequence length - fixed)
            observable_dim: Dimension of each observable step
            hidden_dim: Hidden dimension
            forecast_steps: Number of forecast steps (output sequence length - fixed)
                Note: If number of steps is 1 then this model can be used as autoregressive
            n_layers: Number of Linear layers
        """
        super().__init__()
        self._forecast_steps = forecast_steps
        self._flatten = nn.Flatten()
        self._mlp = MLP(
            input_dim=observable_steps*(observable_dim+1),  # time dimension
            hidden_dim=hidden_dim,
            output_dim=forecast_steps * observable_dim,
            n_layers=n_layers
        )

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None) -> torch.Tensor:
        _ = t_unobs  # Unused
        batch_size = x.shape[1]

        # Add relative time points to input data
        xt = torch.cat([x, t_obs], dim=-1)

        xt = torch.permute(xt, [1, 0, 2])  # (time, batch, bbox) -> (batch, time, bbox)
        xt = self._flatten(xt)  # (batch, time, bbox) -> (batch, time*bbox)
        output = self._mlp(xt)  # (batch, steps*bbox)
        output = output.view(batch_size, self._forecast_steps, -1)  # (batch, steps*bbox) -> (batch, steps, bbox)
        output = torch.permute(output, [1, 0, 2])  # (batch, steps, bbox) -> (steps, batch, bbox)
        return output


class LightningMLPForecaster(LightningModuleForecaster):
    """
    MLPForecaster trainer wrapper.
    """
    def __init__(
        self,
        observable_steps: int,
        observable_dim: int,
        hidden_dim: int,
        forecast_steps: int,
        n_layers: int = 1,

        train_config: Optional[LightningTrainConfig] = None,
        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None,
    ):
        """
        Args:
            observable_steps: Number of observable steps (input sequence length - fixed)
            observable_dim: Dimension of each observable step
            hidden_dim: Hidden dimension
            forecast_steps: Number of forecast steps (output sequence length - fixed)
                Note: If number of steps is 1 then this model can be used as autoregressive
            n_layers: Number of Linear layers
            train_config: TrainConfig (not required for inference)
        """
        model = MLPForecaster(
            observable_steps=observable_steps,
            observable_dim=observable_dim,
            hidden_dim=hidden_dim,
            forecast_steps=forecast_steps,
            n_layers=n_layers
        )
        super().__init__(
            train_config=train_config,
            model=model,
            transform_func=transform_func
        )


# noinspection DuplicatedCode
def main() -> None:
    model = LightningMLPForecaster(
        observable_steps=10,
        observable_dim=5,
        hidden_dim=3,
        forecast_steps=20,
        n_layers=2
    )
    xs = torch.randn(10, 3, 5)
    ts_obs = torch.randn(10, 3, 1)
    ts_unobs = torch.rand(20, 3, 1)
    output = model(xs, ts_obs, ts_unobs)
    expected_shape = (20, 3, 5)

    assert output.shape == expected_shape, f'Expected shape {expected_shape} but found {output.shape}!'


if __name__ == '__main__':
    main()
