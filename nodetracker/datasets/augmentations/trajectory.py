"""
Implementation of trajectory augmentations
"""
from abc import abstractmethod, ABC
from typing import Tuple, List
import random

import torch


class TrajectoryAugmentation(ABC):
    """
    Abstract augmentation - defines interface
    """
    @abstractmethod
    def apply(self, x_obs: torch.Tensor, x_unobs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_obs: Observed bboxes data
            x_unobs: Unobserved bboxes data
            t_obs: Observed time points
            t_unobs: Unobserved time points

        Returns:
            augmented (x_obs, x_unobs, t_obs, t_unobs)
        """
        pass

    def __call__(self, x_obs: torch.Tensor, x_unobs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Alias for `apply` function.

        Args:
            x_obs: Observed bboxes data
            x_unobs: Unobserved bboxes data
            t_obs: Observed time points
            t_unobs: Unobserved time points

        Returns:
            augmented (x_obs, t_obs, x_unobs, t_unobs)
        """
        return self.apply(x_obs, x_unobs, t_obs, t_unobs)


class IdentityAugmentation(TrajectoryAugmentation):
    """
    Performs no transformations (identity).
    """
    def apply(self, x_obs: torch.Tensor, x_unobs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return x_obs, x_unobs, t_obs, t_unobs


class CompositionAugmentation(TrajectoryAugmentation):
    """
    Composition of multiple augmentations.
    """
    def __init__(self, augs: List[TrajectoryAugmentation]):
        """
        Args:
            augs: List of augmentations
        """
        self._augs = augs

    def apply(self, x_obs: torch.Tensor, x_unobs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for aug in self._augs:
            x_obs, x_unobs, t_obs, t_unobs = aug.apply(x_obs, t_obs, x_unobs, t_unobs)
        return x_obs, x_unobs, t_obs, t_unobs


class DetectorNoiseAugmentation(TrajectoryAugmentation):
    """
    Add Gaussian noise based on the bbox width and height.
    """
    def __init__(self, sigma: float = 0.05, proba: float = 0.5):
        """
        Args:
            sigma: Noise multiplier
            proba: Probability to apply this augmentation
        """
        self._sigma = sigma
        self._proba = proba

    def apply(self, x_obs: torch.Tensor, x_unobs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r = random.uniform(0, 1)
        if r > self._proba:
            # Skip augmentation
            return x_obs, x_unobs, t_obs, t_unobs

        x_obs_noise = self._sigma * torch.randn_like(x_obs)
        if len(x_obs.shape) == 3:  # Batch operation
            x_obs_noise[:, :, 0] *= x_obs[:, :, 2]  # `x` noise is proportional to the `h`
            x_obs_noise[:, :, 2] *= x_obs[:, :, 2]  # `h` noise is proportional to the `h`
            x_obs_noise[:, :, 1] *= x_obs[:, :, 3]  # `y` noise is proportional to the `w`
            x_obs_noise[:, :, 3] *= x_obs[:, :, 3]  # `w` noise is proportional to the `w`
        elif len(x_obs.shape) == 2:  # Instance operation
            x_obs_noise[:, 0] *= x_obs[:, 2]  # `x` noise is proportional to the `h`
            x_obs_noise[:, 2] *= x_obs[:, 2]  # `h` noise is proportional to the `h`
            x_obs_noise[:, 1] *= x_obs[:, 3]  # `y` noise is proportional to the `w`
            x_obs_noise[:, 3] *= x_obs[:, 3]  # `w` noise is proportional to the `w`
        else:
            raise AssertionError(f'DetectorNoiseAugmentation supports tensors 2D and 3D tensors only. Got {x_obs.shape}')

        x_obs += x_obs_noise
        return x_obs, x_unobs, t_obs, t_unobs

class ShortenTrajectoryAugmentation(TrajectoryAugmentation):
    """
    Shortens the input trajectory.
    """
    def __init__(self, min_length: int, proba: float):
        """
        Args:
            min_length: Min Trajectory length
                - Augmented trajectory can't be shortened
                  if it already has length `min_length` or less
            proba: Probability to apply
        """
        self._min_length = min_length
        self._proba = proba

    def apply(self, x_obs: torch.Tensor, x_unobs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r = random.uniform(0, 1)
        if r > self._proba:
            # Skip augmentation
            return x_obs, x_unobs, t_obs, t_unobs

        traj_length = x_obs.shape[0]
        if traj_length <= self._min_length:
            return x_obs, x_unobs, t_obs, t_unobs

        new_traj_length = round(random.uniform(self._min_length, traj_length))
        traj_start = traj_length - new_traj_length
        x_obs = x_obs[traj_start:, ...]
        t_obs = t_obs[traj_start:, ...]

        return x_obs, x_unobs, t_obs, t_unobs


def create_identity_augmentation_config() -> dict:
    """
    Returns:
        Identity augmentation (no transformation) config
    """
    return {
        '_target_': 'nodetracker.datasets.augmentations.trajectory.IdentityAugmentation',
    }
