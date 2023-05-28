import random
from typing import Optional

import torch


def simulate_detector_noise(
    measurement: torch.Tensor,
    prev_measurement: Optional[torch.Tensor],
    sigma: float
) -> torch.Tensor:
    """
    Adds measurements noise (random variable) to measurement.
    Noise covariance matrix for step k is a diagonal matrix with values:
    [ (sigma * h[k-1])^2, (sigma * w[k-1])^2, (sigma * h[k-1])^2, (sigma * w[k-1])^2 ]

    Noise is time-dependant on previous bbox width and height.

    Args:
        measurement: Current measurement (ground truth)
        prev_measurement: Previous measurement
        sigma: Noise multiplier

    Returns:
        Measurement with added noise
    """
    if prev_measurement is None:
        return measurement  # First measurement does not have any noise
    x, y, h, w = measurement
    _, _, p_h, p_w = prev_measurement
    x_noise, h_noise = sigma * torch.randn(2) * p_h
    y_noise, w_noise = sigma * torch.randn(2) * p_w

    return torch.tensor([
        x + x_noise,
        y + y_noise,
        h + h_noise,
        w + w_noise
    ], dtype=measurement.dtype)


def simulate_detector_false_positive(proba: float, first_frame: bool) -> bool:
    """
    "Deletes" measurement with some probability. First frame can't be skipped.

    Args:
        proba: Probability to skip detection
        first_frame: Is it first frame for the object

    Returns:
        True if detection is skipped else False
    """
    if first_frame:
        return False

    r = random.uniform(0, 1)
    return r < proba
