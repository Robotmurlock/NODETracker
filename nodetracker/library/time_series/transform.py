"""
Set of time-series transformation functions
"""
import torch


def first_difference(xs: torch.Tensor) -> torch.Tensor:
    """
    Transforms time series to first difference. Let X be the data stream and Y the result:
        Y[i] = X[i] - X[i-1] if i > 0
        Y[0] = 0

    Args:
        xs: Time-series (data stream)

    Returns:
        First difference of the time-series
    """
    diff = torch.zeros_like(xs)
    diff[0] = 0
    diff[1:] = xs[1:] - xs[:-1]
    diff = diff.to(xs)
    return diff
