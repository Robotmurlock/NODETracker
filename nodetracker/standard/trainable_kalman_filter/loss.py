import torch
from torch import nn


class LinearGaussianEnergyFunction(nn.Module):
    """
    Energy (loss) function for linear Gaussian model (e.g. Kalman filter)

    References:
        - Energy function for linear Gaussian model
            https://users.aalto.fi/%7Essarkka/pub/cup_book_online_20131111.pdf (page 187)
        - Pytorch GaussianNLLoss
            https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
            Note: Pytorch implementation supports only diagonal covariance matrix
    """
    def __init__(self, eps: float = 1e-6, reduction: str = 'none'):
        """
        Args:
            eps: Clamping variance for stability.
        """
        super().__init__()
        self._eps = eps
        self._reduction = reduction.lower()

        reduction_options = ['mean', 'sum', 'none']
        if self._reduction not in reduction_options:
            raise ValueError(f'Invalid reduction option "{reduction_options}". Valid options {reduction_options}.')

    def forward(self, innovation: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        P = torch.clamp(P, min=self._eps)
        loss = (1 / 2) * torch.log(2 * torch.pi * P) + (1 / 2) * innovation.T @ torch.inverse(P) @ innovation

        if self._reduction == 'mean':
            return loss.mean()
        if self._reduction == 'sum':
            return loss.sum()
        if self._reduction == 'none':
            return loss

        raise AssertionError(f'Invalid Program State! Reduction option: "{self._reduction}".')
