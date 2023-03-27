"""
Implementation of trajectory augmentations
"""
from abc import abstractmethod, ABC
from typing import Tuple, List

import torch


class TrajectoryAugmentation(ABC):
    """
    Abstract augmentation - defines interface
    """
    @abstractmethod
    def apply(self, x_obs: torch.Tensor, t_obs: torch.Tensor, x_unobs: torch.Tensor, t_unobs: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_obs: Observed bboxes data
            t_obs: Observed time points
            x_unobs: Unobserved bboxes data
            t_unobs: Unobserved time points

        Returns:
            augmented (x_obs, t_obs, x_unobs, t_unobs)
        """
        pass

    def __call__(self, x_obs: torch.Tensor, t_obs: torch.Tensor, x_unobs: torch.Tensor, t_unobs: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Alias for `apply` function.

        Args:
            x_obs: Observed bboxes data
            t_obs: Observed time points
            x_unobs: Unobserved bboxes data
            t_unobs: Unobserved time points

        Returns:
            augmented (x_obs, t_obs, x_unobs, t_unobs)
        """
        return self.apply(x_obs, t_obs, x_unobs, t_unobs)


class IdentityAugmentation(TrajectoryAugmentation):
    """
    Performs no transformations (identity).
    """
    def apply(self, x_obs: torch.Tensor, t_obs: torch.Tensor, x_unobs: torch.Tensor, t_unobs: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return x_obs, t_obs, x_unobs, t_unobs


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

    def apply(self, x_obs: torch.Tensor, t_obs: torch.Tensor, x_unobs: torch.Tensor, t_unobs: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for aug in self._augs:
            x_obs, t_obs, x_unobs, t_unobs = aug.apply(x_obs, t_obs, x_unobs, t_unobs)
        return x_obs, t_obs, x_unobs, t_unobs


class DetectorNoiseAugmentation(TrajectoryAugmentation):
    """
    Add Gaussian noise based on the bbox width and height.
    """
    def __init__(self, sigma: float = 0.05):
        """
        Args:
            sigma: Noise multiplier
        """
        self._sigma = sigma

    def apply(self, x_obs: torch.Tensor, t_obs: torch.Tensor, x_unobs: torch.Tensor, t_unobs: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return x_obs, t_obs, x_unobs, t_unobs


def create_identity_augmentation_config() -> dict:
    """
    Returns:
        Identity augmentation (no transformation) config
    """
    return {
        '_target_': 'nodetracker.datasets.augmentations.trajectory.IdentityAugmentation',
    }

