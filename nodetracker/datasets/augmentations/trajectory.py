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
    def apply(self, x_obs: torch.Tensor, x_unobs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def apply(self, x_obs: torch.Tensor, x_unobs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for aug in self._augs:
            x_obs, x_unobs, t_obs, t_unobs = aug.apply(x_obs, x_unobs, t_obs, t_unobs)
        return x_obs, x_unobs, t_obs, t_unobs



class NonDeterministicAugmentation(TrajectoryAugmentation, ABC):
    def __init__(self, proba: float):
        self._proba = proba

    def apply(self, x_obs: torch.Tensor, x_unobs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r = random.uniform(0, 1)
        if r > self._proba:
            # Skip augmentation
            return x_obs, x_unobs, t_obs, t_unobs

        return self._apply(x_obs, x_unobs, t_obs, t_unobs)

    @abstractmethod
    def _apply(self, x_obs: torch.Tensor, x_unobs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) \
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


class DetectorNoiseAugmentation(NonDeterministicAugmentation):
    """
    Add Gaussian noise based on the bbox width and height.
    """
    def __init__(self, sigma: float = 0.05, proba: float = 0.5, unobs_noise: bool = False):
        """
        Args:
            sigma: Noise multiplier
            proba: Probability to apply this augmentation
            unobs_noise: Apply noise to unobserved part of trajectory
        """
        super().__init__(proba=proba)
        self._sigma = sigma
        self._unobs_noise = unobs_noise

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds guassian vector to bboxes vector x.

        Args:
            x:

        Returns:
            Input vector with noise
        """
        x_noise = self._sigma * torch.randn_like(x)
        x_noise[..., 0] *= x[..., 2]  # `x` noise is proportional to the `h`
        x_noise[..., 2] *= x[..., 2]  # `h` noise is proportional to the `h`
        x_noise[..., 1] *= x[..., 3]  # `y` noise is proportional to the `w`
        x_noise[..., 3] *= x[..., 3]  # `w` noise is proportional to the `w`
        return x + x_noise


    def _apply(self, x_obs: torch.Tensor, x_unobs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_obs = self._add_noise(x_obs)
        if self._unobs_noise:
            x_unobs = self._add_noise(x_unobs)

        return x_obs, x_unobs, t_obs, t_unobs


class ShortenTrajectoryAugmentation(NonDeterministicAugmentation):
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
        super().__init__(proba=proba)
        self._min_length = min_length

    def _apply(self, x_obs: torch.Tensor, x_unobs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        traj_length = x_obs.shape[0]
        if traj_length <= self._min_length:
            return x_obs, x_unobs, t_obs, t_unobs

        new_traj_length = round(random.uniform(self._min_length, traj_length))
        traj_start = traj_length - new_traj_length
        x_obs = x_obs[traj_start:, ...]
        t_obs = t_obs[traj_start:, ...]

        return x_obs, x_unobs, t_obs, t_unobs


def remove_points(xs: List[torch.Tensor], min_length: int) -> Tuple[List[torch.Tensor], List[int], List[int]]:
    """
    Helper function that removes random number of points from trajectory.

    Args:
        xs: Trajectory components (e.g. values, time, ...)
            - Same operation is performed on each of these tensors
            - It is assumed that their first dimension is same and that they are 3D
        min_length: Minimum result trajectory length

    Returns:
        Trajectory with some removed points, List of removed points, List of kept points
    """
    n_points = xs[0].shape[0]
    max_points_to_remove = max(0, n_points - min_length)
    n_points_to_remove = random.randrange(0, max_points_to_remove)
    all_point_indices = list(range(n_points))
    points_to_remove = random.sample(all_point_indices, k=n_points_to_remove)
    points_to_keep = [point for point in all_point_indices if point not in points_to_remove]

    return [x[points_to_keep, :, :] for x in xs], points_to_remove, points_to_keep


class RemoveRandomPointsTrajectoryAugmentation(NonDeterministicAugmentation):
    """
    Remove random points from input trajectory.
    """
    def __init__(self, min_length: int, proba: float):
        """
        Args:
            min_length: Min Trajectory length
                - Augmented trajectory can't be shortened more than `min_length`
            proba: Probability to apply
        """
        super().__init__(proba=proba)
        self._min_length = min_length

    def _apply(self, x_obs: torch.Tensor, x_unobs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (x_obs, t_obs), _, _ = remove_points([x_obs, t_obs], self._min_length)
        return x_obs, x_unobs, t_obs, t_unobs


class RemoveRandomPointsUnobservedTrajectoryAugmentation(NonDeterministicAugmentation):
    """
    Remove random points from target trajectory. Same as `RemoveRandomPointsTrajectoryAugmentation` but for unobserved trajectory.
    """
    def __init__(self, min_length: int, proba: float):
        """
        Args:
            min_length: Min Trajectory length
                - Augmented trajectory can't be shortened more than `min_length`
            proba: Probability to apply
        """
        super().__init__(proba=proba)
        self._min_length = min_length
        self._proba = proba
        assert False, 'This augmentation is currently not supported!'

    def _apply(self, x_obs: torch.Tensor, x_unobs: torch.Tensor, t_obs: torch.Tensor, t_unobs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (x_unobs, t_unobs), _, _ = remove_points([x_unobs, t_unobs], self._min_length)
        return x_obs, x_unobs, t_obs, t_unobs


def create_identity_augmentation_config() -> dict:
    """
    Returns:
        Identity augmentation (no transformation) config
    """
    return {
        '_target_': 'nodetracker.datasets.augmentations.trajectory.IdentityAugmentation',
    }
