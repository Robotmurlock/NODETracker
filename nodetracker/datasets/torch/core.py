"""
Torch dataset support.
Any Dataset that implements `TrajectoryDataset` interface can be used for training and evaluation.
"""
from abc import abstractmethod, ABC
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset

from nodetracker.datasets import augmentations, transforms


class TrajectoryDataset(ABC):
    """
    Defines interface for TrajectoryDataset.
    """
    def __init__(self, history_len: int, future_len: int, sequence_list: Optional[List[str]] = None, **kwargs):
        self._history_len = history_len
        self._future_len = future_len
        self._sequence_list = sequence_list

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def scenes(self) -> List[str]:
        """
        Returns:
            List of scenes in dataset.
        """

    @abstractmethod
    def parse_object_id(self, object_id: str) -> Tuple[str, str]:
        """
        Parses and validates object id.

        Object id convention is `{scene_name}_{scene_object_id}` and is unique over all scenes.

        For MOT {scene_name} represents one video sequence.
        For SOT {scene_name} does not need to be unique for the sequence
        but `{scene_name}_{scene_object_id}` is always unique

        Args:
            object_id: Object id

        Returns:
            scene name, scene object id
        """

    @abstractmethod
    def get_scene_object_ids(self, scene_name: str) -> List[str]:
        """
        Gets object ids for given scene name

        Args:
            scene_name: Scene name

        Returns:
            Scene objects
        """

    def get_scene_number_of_object_ids(self, scene_name: str) -> int:
        """
        Gets number of unique objects in the scene.

        Args:
            scene_name: Scene name

        Returns:
            Number of objects in the scene
        """
        return len(self.get_scene_object_ids(scene_name))

    @abstractmethod
    def get_object_data_length(self, object_id: str) -> int:
        """
        Gets total number of data points for given `object_id` for .

        Args:
            object_id: Object id

        Returns:
            Number of data points
        """

    @abstractmethod
    def get_object_data_label(self, object_id: str, index: int, relative_bbox_coords: bool = True) -> dict:
        """
        Get object data point index.

        Args:
            object_id: Object id
            index: Index
            relative_bbox_coords: Scale bbox coords to [0, 1]

        Returns:
            Data point.
        """

    @abstractmethod
    def get_object_data_label_by_frame_index(
        self,
        object_id: str,
        frame_index: int,
        relative_bbox_coords: bool = True
    ) -> Optional[dict]:
        """
        Like `get_object_data_label` but data is relative to given frame_index.
        If object does not exist in given frame index then None is returned.

        Args:
            object_id: Object id
            frame_index: Frame Index
            relative_bbox_coords: Scale bbox coords to [0, 1]

        Returns:
            Data point.
        """

    @abstractmethod
    def get_scene_info(self, scene_name: str) -> Any:
        """
        Get scene metadata by name.

        Args:
            scene_name: Scene name

        Returns:
            Scene metadata
        """

    @abstractmethod
    def get_scene_image_path(self, scene_name: str, frame_id: int) -> str:
        """
        Get image (frame) path for given scene and frame id.

        Args:
            scene_name: scene name
            frame_id: frame id

        Returns:
            Frame path
        """


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

    def __getitem__(self, index: int) \
            -> Tuple[torch.tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata = self._dataset[index]
        orig_bboxes_obs = torch.from_numpy(bboxes_obs)
        bboxes_unobs = torch.from_numpy(bboxes_unobs)
        ts_obs = torch.from_numpy(ts_obs)
        ts_unobs = torch.from_numpy(ts_unobs)

        # Trajectory transformations
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs = \
            self._augmentation_before_transform(orig_bboxes_obs, bboxes_unobs, ts_obs, ts_unobs)

        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata, _ = \
            self._transform([bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata, None], shallow=False)

        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs = \
            self._augmentation_after_transform(bboxes_obs, bboxes_unobs, ts_obs, ts_unobs)

        return bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, orig_bboxes_obs, metadata
