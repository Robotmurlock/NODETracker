"""
Helper classes and function for MLP-ResNet blocks.
"""
import torch
from torch import nn

from nodetracker.library.building_blocks.mlp import MLP


class ResnetMLPBlock(nn.Module):
    """
    Like `MLP` but with residual (skip connections). Uses `MLP` as a bottleneck default layer.
    """
    def __init__(
        self,
        dim: int,
        n_layers: int = 2,
        n_bottleneck_layers: int = 1,
        lrelu_slope: float = 1e-2
    ):
        """
        Args:
            dim: Model Input/Hidden/Output dimension
            n_layers: Number of Perceptron layers
            n_bottleneck_layers: Number of layers in bottleneck branch
            lrelu_slope: LeakyReLU slope
        """
        super().__init__()
        assert n_layers >= 1, f'Minimum number of layers is 1 but found {n_layers}'

        self._bottleneck = MLP(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            n_layers=n_bottleneck_layers,
            lrelu_slope=lrelu_slope
        )
        self._relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        x = self._bottleneck(x)
        return self._relu(x + x_skip)


def run_test() -> None:
    rmb = ResnetMLPBlock(4, n_layers=4)
    x = torch.randn(32, 4)

    print(rmb(x).shape)


if __name__ == '__main__':
    run_test()
