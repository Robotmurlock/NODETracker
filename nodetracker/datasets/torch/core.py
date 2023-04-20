from typing import Optional, Tuple, Dict, Any
from abc import abstractmethod, ABC
import numpy as np

import torch
from torch.utils.data import Dataset

from nodetracker.datasets import augmentations, transforms


class TrajectoryDataset(ABC):
    """
    Defines interface for TrajectoryDataset.
    """
    def __init__(self, history_len: int, future_len: int):
        self._history_len = history_len
        self._future_len = future_len

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class TorchTrajectoryDataset(Dataset):
    """
    PyTorch wrapper for Trajectory dataset
    """
    def __init__(
        self,
        dataset: TrajectoryDataset,
        transform: Optional[transforms.InvertibleTransform] = None,
        augmentation_before_transform: Optional[augmentations.TrajectoryAugmentation] = None,
        augmentation_after_transform: Optional[augmentations.TrajectoryAugmentation] = None
    ) -> None:
        """
        Args:
            dataset: Dataset that implements `__len__` and `__getitem`
            transform: Item postprocess
        """
        super().__init__()
        self._dataset = dataset

        self._transform = transform
        if self._transform is None:
            self._transform = transforms.IdentityTransform()

        self._augmentation_before_transform = augmentation_before_transform
        if self._augmentation_before_transform is None:
            self._augmentation_before_transform = augmentations.IdentityAugmentation()

        self._augmentation_after_transform = augmentation_after_transform
        if self._augmentation_after_transform is None:
            self._augmentation_after_transform = augmentations.IdentityAugmentation()

    @property
    def dataset(self) -> TrajectoryDataset:
        """
        Returns: MOT core dataset class object.
        """
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata = self._dataset[index]
        bboxes_obs = torch.from_numpy(bboxes_obs)
        bboxes_unobs = torch.from_numpy(bboxes_unobs)
        ts_obs = torch.from_numpy(ts_obs)
        ts_unobs = torch.from_numpy(ts_unobs)

        # Trajectory transformations
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs = \
            self._augmentation_before_transform(bboxes_obs, bboxes_unobs, ts_obs, ts_unobs)

        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata = \
            self._transform([bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata])

        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs = \
            self._augmentation_after_transform(bboxes_obs, bboxes_unobs, ts_obs, ts_unobs)

        return bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata