import torch
import torch.linalg as LA
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
    def __init__(
        self,
        optimize_likelihood: bool = True,
        eps: float = 1e-6,
        reduction: str = 'mean'
    ):
        """
        Args:
            optimize_likelihood: If False then MSE is used instead of EnergyFunction (improvisation)
            eps: Clamping variance for stability.
            reduction: Loss reduction option
        """
        super().__init__()
        self._optimize_likelihood = optimize_likelihood
        self._eps = eps
        self._reduction = reduction.lower()

        reduction_options = ['mean', 'sum', 'none']
        if self._reduction not in reduction_options:
            raise ValueError(f'Invalid reduction option "{reduction_options}". Valid options {reduction_options}.')

    def forward(self, innovation: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        P = torch.clamp(P, min=self._eps)
        innovation_T = torch.transpose(innovation, dim0=-2, dim1=-1)

        if self._optimize_likelihood:
            loss = (1 / 2) * torch.log(torch.norm(2 * torch.pi * P)) \
                   + (1 / 2) * torch.bmm(torch.bmm(innovation_T, LA.pinv(P)), innovation)
        else:
            loss = torch.bmm(innovation_T, innovation)

        if self._reduction == 'mean':
            return loss.mean()
        if self._reduction == 'sum':
            return loss.sum()
        if self._reduction == 'none':
            return loss

        raise AssertionError(f'Invalid Program State! Reduction option: "{self._reduction}".')
