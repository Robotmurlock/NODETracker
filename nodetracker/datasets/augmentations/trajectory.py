"""
Implementation of trajectory augmentations
"""
from abc import abstractmethod, ABC
from nodetracker.utils.serialization import Serializable, JSONType
import torch
from typing import Tuple, List, Any


class TrajectoryAugmentation(Serializable, ABC):
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

    def serialize(self) -> JSONType:
        return {
            '_target_': 'nodetracker.datasets.augmentations.trajectory.DetectorNoiseAugmentation',  # TODO
            'augs': [v.serialize() for v in self._augs]
        }

    @classmethod
    def deserialize(cls, raw: JSONType) -> Any:
        del raw['_target_']  # Redundant
        return cls(**raw)


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
        x_obs_noise[:, :, [0, 2]] *= x_obs[:, :, 2]  # `x` and `h` noise are proportional to the `h`
        x_obs_noise[:, :, [1, 3]] *= x_obs[:, :, 3]  # `y` and `w` noise are proportional to the `w`
        x_obs += x_obs_noise
        return x_obs, t_obs, x_unobs, t_unobs

    def serialize(self) -> JSONType:
        return {
            '_target_': 'nodetracker.datasets.augmentations.trajectory.DetectorNoiseAugmentation',  # TODO
            'sigma': self._sigma
        }

    @classmethod
    def deserialize(cls, raw: JSONType) -> Any:
        del raw['_target_']  # Redundant
        return cls(**raw)


def create_identity_augmentation() -> TrajectoryAugmentation:
    """
    Returns:
        Creates identity augmentation (no transformation)
    """
    return CompositionAugmentation(augs=[])
