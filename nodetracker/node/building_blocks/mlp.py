"""
Building block: MultiLayerPerceptron
"""
import torch
from torch import nn
from typing import Optional


class MLP(nn.Module):
    """
    MultiLayerPerceptron where each linear layer contains:
    - Linear layer
    - LayerNorm
    - LeakyReLU activation
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        n_layers: int = 1,
        lrelu_slope: float = 1e-2
    ):
        """
        Args:
            input_dim: Model Input dimension
            hidden_dim: Model hidden dimension
                - tensor dimension between layers
                - equal to `input_dim` if not defined
                - not relevant in case `n_layers` is equal to `1`
            output_dim: Model output dimension
                - equal to `input_dim` if not defined
            n_layers: Number of Perceptron layers
            lrelu_slope: LeakyReLU slope
        """
        super().__init__()
        assert n_layers >= 1, f'Minimum number of layers is 1 but found {n_layers}'
        if output_dim is None:
            output_dim = input_dim
        if hidden_dim is None:
            hidden_dim = input_dim

        layers_args = [[hidden_dim, hidden_dim] for _ in range(n_layers)]
        layers_args[0][0] = input_dim
        layers_args[-1][1] = output_dim
        self._model = nn.Sequential(*[self._create_mlp_layer(*args, lrelu_slope=lrelu_slope) for args in layers_args])

    @staticmethod
    def _create_mlp_layer(input_dim: int, output_dim: int, lrelu_slope: float = 1e-2) -> nn.Module:
        """
        Constructs previously defined MLP layer

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            lrelu_slope: LeakyReLU slope

        Returns:
            Perceptron
        """
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(lrelu_slope)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


def main() -> None:
    mlp = MLP(
        input_dim=5,
        hidden_dim=10,
        n_layers=2
    )

    x = torch.randn(32, 5)  # (batch_size, dim)
    print('Output shape:', mlp(x).shape)


if __name__ == '__main__':
    main()