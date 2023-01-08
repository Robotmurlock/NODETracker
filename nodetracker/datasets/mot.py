"""
MOT Challenge Dataset support
"""
import configparser
import enum
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from nodetracker.datasets.utils import ode_dataloader_collate_func
from nodetracker.utils.logging import configure_logging


class LabelType(enum.Enum):
    DETECTION = 'det'
    GROUND_TRUTH = 'gt'


@dataclass
class SceneInfo:
    """
    MOT Scene metadata (name, frame shape, ...)
    """
    name: str
    dirpath: str
    gt_path: str
    seqlength: Union[str, int]
    framerate: Union[str, int]
    imdir: str
    imext: str
    imheight: Union[str, int]
    imwidth: Union[str, int]

    def __post_init__(self):
        """
        Convert to proper type
        """
        self.seqlength = int(self.seqlength)
        self.framerate = int(self.framerate)
        self.imheight = int(self.imheight)
        self.imwidth = int(self.imwidth)


SceneInfoIndex = Dict[str, SceneInfo]


logger = logging.getLogger('MOTDataset')

class MOTDataset:
    """
    Parses MOT dataset in given format
    """
    def __init__(self, path: str, history_len: int, future_len: int, label_type: LabelType = LabelType.GROUND_TRUTH) -> None:
        """
        Args:
            path: Path to dataset
            history_len: Number of observed data points
            future_len: Number of unobserved data points
            label_type: Label Type
        """
        self._path = path
        self._history_len = history_len
        self._future_len = future_len
        self._label_type = label_type

        self._scene_info_index = self._index_dataset(path, label_type)

        self._data_labels, self._n_labels = self._parse_labels(self._scene_info_index)
        self._trajectory_index = self._create_trajectory_index(self._data_labels, self._history_len, self._future_len)

    @property
    def label_type(self) -> LabelType:
        """
        Returns: LabelType
        """
        return self._label_type

    def get_scene_info(self, scene_name: str) -> SceneInfo:
        """
        Get scene metadata by name.

        Args:
            scene_name: Scene name

        Returns:
            Scene metadata
        """
        return self._scene_info_index[scene_name]

    @staticmethod
    def _get_image_path(scene_info: SceneInfo, frame_id: int) -> str:
        """
        Get frame path for given scene and frame id

        Args:
            scene_info: scene metadata
            frame_id: frame number

        Returns:
            Path to image (frame)
        """
        return os.path.join(scene_info.dirpath, scene_info.imdir, f'{frame_id:06d}{scene_info.imext}')

    def get_scene_image_path(self, scene_name: str, frame_id: int) -> str:
        """
        Get image (frame) path for given scene and frame id.

        Args:
            scene_name: scene name
            frame_id: frame id

        Returns:
            Frame path
        """
        scene_info = self._scene_info_index[scene_name]
        return self._get_image_path(scene_info, frame_id)

    @staticmethod
    def _index_dataset(path: str, label_type: LabelType) -> SceneInfoIndex:
        """
        Index dataset content. Format: { {scene_name}: {scene_labels_path} }

        Args:
            path: Path to dataset
            label_type: Use ground truth bboxes or detections

        Returns:
            Index to scenes
        """
        scene_names = [file for file in os.listdir(path) if not file.startswith('.')]
        logger.debug(f'Found {len(scene_names)} scenes. Names: {scene_names}.')

        scene_info_index: SceneInfoIndex = {}

        for scene_name in scene_names:
            scene_directory = os.path.join(path, scene_name)
            scene_files = os.listdir(scene_directory)
            assert label_type.value in scene_files, f'Ground truth file "{label_type.value}" not found. Contents: {scene_files}'
            gt_path = os.path.join(scene_directory, label_type.value, f'{label_type.value}.txt')

            assert 'seqinfo.ini' in scene_files, f'Scene config file "seqinfo.ini" not found. Contents: {scene_files}'
            scene_info_path = os.path.join(scene_directory, 'seqinfo.ini')
            raw_info = configparser.ConfigParser()
            raw_info.read(scene_info_path)
            raw_info = dict(raw_info['Sequence'])
            raw_info['gt_path'] = gt_path
            raw_info['dirpath'] = scene_directory

            scene_info = SceneInfo(**raw_info)
            scene_info_index[scene_name] = scene_info
            logger.debug(f'Scene info {scene_info}.')

        return scene_info_index

    @staticmethod
    def _parse_labels(scene_infos: SceneInfoIndex) -> Tuple[Dict[str, list], int]:
        """
        Loads all labels dictionary with format:
        {
            {scene_name}_{object_id}: {
                {frame_id}: [ymin, xmin, w, h]
            }
        }

        Args:
            scene_infos: Scene Metadata

        Returns:
            Labels dictionary
        """
        data = defaultdict(list)
        n_labels = 0

        for scene_name, scene_info in scene_infos.items():
            df = pd.read_csv(scene_info.gt_path, header=None)
            df = df[df[7] == 1]  # Ignoring non-pedestrian objects

            df = df.iloc[:, :6]
            df.columns = ['frame_id', 'object_id', 'ymin', 'xmin', 'w', 'h']
            df['object_global_id'] = scene_name + '_' + df['object_id'].astype(str)  # object id is not unique over all scenes
            df = df.drop(columns='object_id', axis=1)
            df = df.sort_values(by=['object_global_id', 'frame_id'])
            n_labels += df.shape[0]

            object_groups = df.groupby('object_global_id')
            for object_global_id, df_grp in tqdm(object_groups, desc=f'Parsing {scene_name}', unit='pedestrian'):
                df_grp = df_grp.drop(columns='object_global_id', axis=1).set_index('frame_id')
                assert df_grp.index.max() - df_grp.index.min() + 1 == df_grp.index.shape[0], f'Object {object_global_id} has missing data points!'

                for frame_id, row in df_grp.iterrows():
                    data[object_global_id].append({
                        'frame_id': frame_id,
                        'bbox': row.values.tolist(),
                        'image_path': MOTDataset._get_image_path(scene_info, frame_id)
                    })

        logger.debug(f'Parsed labels. Dataset size is {n_labels}.')
        return data, n_labels

    @staticmethod
    def _create_trajectory_index(labels: Dict[str, list], history_len: int, future_len: int) -> List[Tuple[str, int, int]]:
        """
        Creates trajectory index by going through every object and creating bbox trajectory of consecutive time points

        Args:
            labels: List of bboxes for each object
            history_len: Length of observed part of trajectory
            future_len: Length of unobserved part of trajectory

        Returns:
            Trajectory index
        """

        trajectory_len = history_len + future_len
        traj_index = []

        for object_id, data in tqdm(labels.items(), desc='Creating trajectories!', unit='object'):
            object_trajectory_len = len(data)
            for i in range(object_trajectory_len - trajectory_len + 1):
                traj_index.append((object_id, i, i+trajectory_len))

        return traj_index

    def __len__(self) -> int:
        return len(self._trajectory_index)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        object_id, traj_start, traj_end = self._trajectory_index[index]
        raw_traj = self._data_labels[object_id][traj_start:traj_end]
        scene_name = object_id.split('_')[0]
        scene_info = self._scene_info_index[scene_name]

        # Metadata
        frame_ids = [item['frame_id'] for item in raw_traj]
        metadata = {
            'scene_name': scene_name,
            'frame_ids': frame_ids,
            'object_id': object_id,
            'image_paths': [item['image_path'] for item in raw_traj]
        }

        # Bboxes
        bboxes = np.array([item['bbox'] for item in raw_traj], dtype=np.float32)
        # Normalize bbox coordinates
        bboxes[:, [0, 2]] /= scene_info.imwidth
        bboxes[:, [1, 3]] /= scene_info.imheight

        # Time points
        frame_ts = np.array(frame_ids, dtype=np.float32)
        frame_ts = frame_ts - frame_ts[0] + 1  # Transforming to relative time values
        frame_ts = np.expand_dims(frame_ts, -1)

        # Observed - Unobserved
        bboxes_obs = bboxes[:self._history_len]
        bboxes_unobs = bboxes[self._history_len:]
        frame_ts_obs = frame_ts[:self._history_len]
        frame_ts_unobs = frame_ts[self._history_len:]

        return bboxes_obs, bboxes_unobs, frame_ts_obs, frame_ts_unobs, metadata


class TorchMOTTrajectoryDataset(Dataset):
    """
    PyTorch wrapper for MOT dataset
    """
    def __init__(self, path: str, history_len: int, future_len: int, label_type: LabelType = LabelType.GROUND_TRUTH) -> None:
        """
        Args:
            path: Path to dataset
            history_len: Number of observed data points
            future_len: Number of unobserved data points
            label_type: Label Type
        """
        super().__init__()
        self._dataset = MOTDataset(
            path=path,
            history_len=history_len,
            future_len=future_len,
            label_type=label_type
        )

    @property
    def dataset(self) -> MOTDataset:
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

        return bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata

def run_test() -> None:
    from nodetracker.common.project import ASSETS_PATH
    configure_logging(logging.DEBUG)

    dataset_path = os.path.join(ASSETS_PATH, 'MOT20', 'train')
    dataset = MOTDataset(dataset_path, history_len=4, future_len=4)

    print(f'Dataset size: {len(dataset)}')
    print(f'Sample example: {dataset[5]}')

    torch_dataset = TorchMOTTrajectoryDataset(dataset_path, history_len=4, future_len=4)

    print(f'Torch Dataset size: {len(torch_dataset)}')
    print(f'Torch sample example shapes: {[x.shape for x in torch_dataset[5][:-1]]}')
    print(f'Torch sample example: {torch_dataset[5]}')

    torch_dataloader = DataLoader(torch_dataset, batch_size=4, collate_fn=ode_dataloader_collate_func)
    for bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata in torch_dataloader:
        print(f'Torch batch sample example shapes: bboxes_obs={bboxes_obs.shape}, bboxes_unobs={bboxes_unobs.shape}, '
              f'ts_obs={ts_obs.shape}, ts_unobs={ts_unobs.shape}')
        print('Torch batch metadata', metadata)

        break


if __name__ == '__main__':
    run_test()
