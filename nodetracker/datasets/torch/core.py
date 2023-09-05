"""
Torch dataset support.
Any Dataset that implements `TrajectoryDataset` interface can be used for training and evaluation.
"""
from abc import abstractmethod, ABC
from typing import Optional, Tuple, Dict, Any, List, Union

import cv2
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset

from nodetracker.datasets import augmentations, transforms


def interpolate_by_fps(
    fps_multiplier: float,
    bboxes_obs: np.ndarray,
    bboxes_unobs: np.ndarray,
    ts_obs: np.ndarray,
    ts_unobs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fps_multiplier == 1:
        return bboxes_obs, bboxes_unobs, ts_obs, ts_unobs

    ts = np.concatenate([ts_obs, ts_unobs]).reshape(-1)
    bboxes = np.concatenate([bboxes_obs, bboxes_unobs])

    # Base interval
    t_start, t_middle, t_end = [round(float(x)) for x in [ts[0], ts_obs[-1, 0], ts[-1]]]
    n_points = t_end - t_start + 1

    # Transformed interval
    if fps_multiplier > 1:
        assert isinstance(fps_multiplier, int) or fps_multiplier.is_integer()
        t_n_points = (round(fps_multiplier) - 1) * (n_points - 1) + n_points
        t_ts = np.linspace(t_start, t_end, num=t_n_points, endpoint=True, dtype=ts.dtype)
    else:
        step = round(1 / fps_multiplier)
        t_ts = np.arange(t_start, t_end, step, dtype=ts.dtype)

    t_bboxes = np.zeros(shape=(t_ts.shape[0], bboxes.shape[-1]), dtype=bboxes.dtype)
    for dim in range(bboxes.shape[-1]):
        cs = CubicSpline(ts, bboxes[:, dim])
        t_bboxes[:, dim] = cs(t_ts)

    # Create new inputs
    mask = (t_ts <= t_middle)
    new_ts_obs = t_ts[mask].reshape(-1, 1)
    new_ts_unobs = t_ts[~mask].reshape(-1, 1)
    new_bboxes_obs = t_bboxes[mask]
    new_bboxes_unobs = t_bboxes[~mask]

    return new_bboxes_obs, new_bboxes_unobs, new_ts_obs, new_ts_unobs


class TrajectoryDataset(ABC):
    """
    Defines interface for TrajectoryDataset.
    """
    def __init__(
        self,
        history_len: int,
        future_len: int,
        sequence_list: Optional[List[str]] = None,
        image_load: bool = False,
        image_shape: Union[None, List[int], Tuple[int, int]] = None,
        image_bgr_to_rgb: bool = True,
        optical_flow: bool = False,
        optical_flow_3d: bool = False,
        optical_flow_only: bool = True,
        **kwargs
    ):
        self._history_len = history_len
        self._future_len = future_len
        self._sequence_list = sequence_list

        # Image configuration
        self._image_load = image_load

        if image_shape is not None:
            assert isinstance(image_shape, tuple) or isinstance(image_shape, list), \
                f'Invalid image shape type "{type(image_shape)}".'
            image_shape = tuple(image_shape)
            assert len(image_shape) == 2, f'Invalid image shape length "{len(image_shape)}"'

        self._image_shape = image_shape
        self._image_bgr_to_rgb = image_bgr_to_rgb

        # Optical flow configuration
        self._optical_flow = optical_flow
        self._optical_flow_3d = optical_flow_3d
        self._optical_flow_only = optical_flow_only

    def load_image(self, path: str) -> np.ndarray:
        """
        Loads image using cv2.

        Args:
            path: Image source path

        Returns:
            Loaded image
        """
        image = cv2.imread(path)
        if self._image_bgr_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self._image_shape is not None:
            image = cv2.resize(image, self._image_shape, interpolation=cv2.INTER_NEAREST)
        return image

    def perform_optical_flow(self, images: List[np.ndarray]) -> List[np.ndarray]:
        flow_result = []

        for prev_rgb, next_rgb in zip(images[:-1], images[1:]):
            prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            ch1 = ang * 180 / np.pi / 2
            ch2 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            if self._optical_flow_3d:
                hsv = np.zeros(shape=(*prev_gray.shape[:2], 3), dtype=ch1.dtype)
                hsv[..., 1] = 255
                hsv[..., 0] = ch1
                hsv[..., 2] = ch2
                flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            else:
                # hsv with a missing (1) channel*
                hsv = np.zeros(shape=(*prev_gray.shape[:2], 2), dtype=ch1.dtype)
                hsv[..., 0] = ch1
                hsv[..., 1] = ch2
                flow = hsv

            if len(flow_result) == 0:
                # First frame is all zeros (filler)
                flow_result.append(np.zeros_like(flow))

            flow_result.append(flow)

        # Transpose flow "images"
        flow_result = [np.transpose(fr, (2, 0, 1)) for fr in flow_result]
        # Normalize flow "images"
        flow_result = [(fr / 255.0) for fr in flow_result]

        return flow_result

    def update_visual_metadata(self, metadata: dict) -> dict:
        if self._image_load:
            # Images
            image_paths = metadata['image_paths'][:self._history_len]
            metadata['raw_images'] = [self.load_image(path) for path in image_paths]
            metadata['images'] = [np.transpose(img, (2, 0, 1)) / 255.0 for img in metadata['raw_images']]
            metadata['images'] = np.stack(metadata['images'])

            # Optical flow
            if self._optical_flow:
                metadata['flow'] = self.perform_optical_flow(metadata['raw_images'])
                metadata['flow'] = np.stack(metadata['flow'])
                metadata['flow'] = np.nan_to_num(metadata['flow'])

            # Cleanup
            metadata.pop('raw_images')
            if self._optical_flow and self._optical_flow_only:
                metadata.pop('images')

        return metadata

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
    def get_object_category(self, object_id: str) -> str:
        """
        Gets category for object.

        Args:
            object_id: Object id

        Returns:
            Object category
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
        augmentation_after_transform: Optional[augmentations.TrajectoryAugmentation] = None,
        fps_multiplier: float = 1,
    ) -> None:
        """
        Args:
            dataset: Dataset that implements `__len__` and `__getitem`
            transform: Item postprocess
        """
        super().__init__()
        self._dataset = dataset
        self._fps_multiplier = fps_multiplier

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

    def __getitem__(self, index: int) -> Dict[str, Union[dict, torch.Tensor]]:
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata = self._dataset[index]
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs = interpolate_by_fps(self._fps_multiplier, bboxes_obs, bboxes_unobs, ts_obs, ts_unobs)

        orig_bboxes_obs = torch.from_numpy(bboxes_obs)
        orig_bboxes_unobs = torch.from_numpy(bboxes_unobs)
        ts_obs = torch.from_numpy(ts_obs)
        ts_unobs = torch.from_numpy(ts_unobs)
        if 'images' in metadata:
            metadata['images'] = torch.from_numpy(metadata['images'])
        if 'flow' in metadata:
            metadata['flow'] = torch.from_numpy(metadata['flow'])

        # Trajectory transformations
        bboxes_obs, aug_bboxes_unobs, ts_obs, ts_unobs = \
            self._augmentation_before_transform(orig_bboxes_obs, orig_bboxes_unobs, ts_obs, ts_unobs)

        t_bboxes_obs, t_aug_bboxes_unobs, t_ts_obs, t_ts_unobs, t_metadata, _ = \
            self._transform([bboxes_obs, aug_bboxes_unobs, ts_obs, ts_unobs, metadata, None], shallow=False)

        _, t_bboxes_unobs, _, _, _, _ = \
            self._transform([bboxes_obs, orig_bboxes_unobs, ts_obs, ts_unobs, metadata, None], shallow=False)

        t_bboxes_obs, t_aug_bboxes_unobs, t_ts_obs, t_ts_unobs = \
            self._augmentation_after_transform(t_bboxes_obs, t_aug_bboxes_unobs, t_ts_obs, t_ts_unobs)

        return {
            't_bboxes_obs': t_bboxes_obs,
            't_aug_bboxes_unobs': t_aug_bboxes_unobs,
            't_ts_obs': t_ts_obs,
            't_ts_unobs': t_ts_unobs,
            'orig_bboxes_obs': orig_bboxes_obs,
            'orig_bboxes_unobs': orig_bboxes_unobs,
            't_bboxes_unobs': t_bboxes_unobs,
            'metadata': metadata
        }


def run_test() -> None:
    ts_obs = np.array([0, 1, 2, 3], dtype=np.float32).reshape(-1, 1)
    ts_unobs = np.array([4, 5], dtype=np.float32).reshape(-1, 1)
    bboxes_obs = np.array([[0, 0, 1, 1], [0.1, 0.1, 1.1, 1.1], [0.2, 0.2, 1.2, 1.2], [0.3, 0.3, 1.3, 1.3]], dtype=np.float32)
    bboxes_unobs = np.array([[0.4, 0.4, 1.4, 1.4], [0.5, 0.5, 1.5, 1.5]], dtype=np.float32)

    new_bboxes_obs, new_bboxes_unobs, new_ts_obs, new_ts_unobs = \
        interpolate_by_fps(1, bboxes_obs, bboxes_unobs, ts_obs, ts_unobs)

    print(f'Multiplier=1 (shape): {new_ts_obs.shape=}, {new_ts_unobs.shape=}, '
          f'{new_bboxes_obs.shape=}, {new_bboxes_unobs.shape=}')
    print(f'Multiplier=1: {new_ts_obs=}, {new_ts_unobs=}, {new_bboxes_obs=}, {new_bboxes_unobs=}')

    new_bboxes_obs, new_bboxes_unobs, new_ts_obs, new_ts_unobs = \
        interpolate_by_fps(2, bboxes_obs, bboxes_unobs, ts_obs, ts_unobs)

    print(f'Multiplier=2 (shape): {new_ts_obs.shape=}, {new_ts_unobs.shape=}, '
          f'{new_bboxes_obs.shape=}, {new_bboxes_unobs.shape=}')
    print(f'Multiplier=2: {new_ts_obs=}, {new_ts_unobs=}, {new_bboxes_obs=}, {new_bboxes_unobs=}')

    new_bboxes_obs, new_bboxes_unobs, new_ts_obs, new_ts_unobs = \
        interpolate_by_fps(0.5, bboxes_obs, bboxes_unobs, ts_obs, ts_unobs)

    print(f'Multiplier=0.5 (shape): {new_ts_obs.shape=}, {new_ts_unobs.shape=}, '
          f'{new_bboxes_obs.shape=}, {new_bboxes_unobs.shape=}')
    print(f'Multiplier=0.5: {new_ts_obs=}, {new_ts_unobs=}, {new_bboxes_obs=}, {new_bboxes_unobs=}')

if __name__ == '__main__':
    run_test()
