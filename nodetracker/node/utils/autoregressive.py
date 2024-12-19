"""
AutoregressiveForecasterDecorator
"""
from typing import Optional, Tuple

import torch
from torch import nn


def extract_mean_and_std(bboxes_unobs_hat: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TODO: Duplicate code

    Helper function for Gaussian model postprocess

    Args:
        bboxes_unobs_hat: Prediction

    Returns:
        bboxes_hat_mean, bboxes_hat_std
    """
    bboxes_unobs_hat = bboxes_unobs_hat.view(*bboxes_unobs_hat.shape[:-1], -1, 2)
    bboxes_unobs_hat_mean = bboxes_unobs_hat[..., 0]
    bboxes_unobs_hat_log_var = bboxes_unobs_hat[..., 1]
    bboxes_unobs_hat_var = torch.exp(bboxes_unobs_hat_log_var)
    bboxes_unobs_hat_std = torch.sqrt(bboxes_unobs_hat_var)

    return bboxes_unobs_hat_mean, bboxes_unobs_hat_std



class AutoregressiveForecasterDecorator(nn.Module):
    """
    Autoregressive module decorator.
    """
    def __init__(self, model: nn.Module, keep_history: bool = False):
        """
        Args:
            model: Core model use for autoregression process
            keep_history: Keep all history point (works only with models that accept variable input length)
        """
        super().__init__()
        self._model = model
        self._keep_history = keep_history

    @torch.no_grad()
    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None, metadata: Optional[dict] = None) -> torch.Tensor:
        result = []

        n_steps = t_unobs.shape[0]
        for t_i in range(n_steps):
            t_unobs_i = t_unobs[t_i:t_i+1, ...]
            x_hat = self._model(x, t_obs, t_unobs_i, metadata)
            if hasattr(self._model, 'is_modeling_gaussian') and self._model.is_modeling_gaussian:
                x_hat, _ = extract_mean_and_std(x_hat)

            result.append(x_hat)

            if self._keep_history:
                x = torch.cat([x, x_hat])
            else:
                x = torch.cat([x[1:], x_hat])
                t_obs = torch.cat([t_obs[1:], t_unobs_i])

        return torch.cat(result)

    @torch.no_grad()
    def inference(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None, metadata: Optional[dict] = None) -> torch.Tensor:
        """
        Alias for `forward` method.

        Args:
            x: Input trajectory
            t_obs: Input trajectory time points
            t_unobs: "Output" trajectory time points

        Returns:
            Model prediction
        """
        return self(x, t_obs, t_unobs, metadata=metadata)


def run_test():
    class Baseline(nn.Module):
        # noinspection PyMethodMayBeStatic
        def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None) -> torch.Tensor:
            _, _ = t_obs, t_unobs  # Ignore
            return x[-1:]

    model = AutoregressiveForecasterDecorator(Baseline())
    xs = torch.randn(4, 3, 7)
    ts_obs = torch.randn(4, 3, 1)
    ts_unobs = torch.randn(2, 3, 1)
    expected_shape = (2, 3, 7)

    output = model(xs, ts_obs, ts_unobs)
    assert output.shape == expected_shape, f'Expected shape {expected_shape} but found {output.shape}!'
    assert all((torch.abs(output[i] - xs[-1]).sum().item() < 1e-6) for i in range(output.shape[0]))


if __name__ == '__main__':
    run_test()
