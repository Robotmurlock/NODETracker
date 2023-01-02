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
from torch.utils.data import Dataset
from tqdm import tqdm

from nodetracker.utils.logging import configure_logging


class LabelType(enum.Enum):
    DETECTION = 'det'
    GROUND_TRUTH = 'gt'


@dataclass
class SceneInfo:
    name: str
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


SceneIndex = Dict[str, str]
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
            future_len: Number of unobserved data poitns
            label_type: Label Type
        """
        self._history_len = history_len
        self._future_len = future_len

        self._scene_file_index, self._scene_info_index = self._index_dataset(path, label_type)
        self._label_type = label_type

        self._data_labels, self._n_labels = self._parse_labels(self._scene_file_index)
        self._trajectory_index = self._create_trajectory_index(self._data_labels, self._history_len, self._future_len)

    @property
    def label_type(self) -> LabelType:
        """
        Returns: LabelType
        """
        return self._label_type

    @staticmethod
    def _index_dataset(path: str, label_type: LabelType) -> Tuple[SceneIndex, SceneInfoIndex]:
        """
        Index dataset content. Format: { {scene_name}: {scene_labels_path} }

        Args:
            path: Path to dataset
            label_type: Use ground truth bboxes or detections

        Returns:
            Index to scenes and scene files
        """
        scene_names = [file for file in os.listdir(path) if not file.startswith('.')]
        logger.debug(f'Found {len(scene_names)} scenes. Names: {scene_names}.')

        index: SceneIndex = {}
        scene_info_index: SceneInfoIndex = {}

        for scene_name in scene_names:
            scene_directory = os.path.join(path, scene_name)
            scene_files = os.listdir(scene_directory)
            assert label_type.value in scene_files, f'Ground truth file "{label_type.value}" not found. Contents: {scene_files}'
            gt_path = os.path.join(scene_directory, label_type.value, f'{label_type.value}.txt')
            index[scene_name] = gt_path

            assert 'seqinfo.ini' in scene_files, f'Scene config file "seqinfo.ini" not found. Contents: {scene_files}'
            scene_info_path = os.path.join(scene_directory, 'seqinfo.ini')
            raw_info = configparser.ConfigParser()
            raw_info.read(scene_info_path)
            scene_info = SceneInfo(**dict(raw_info['Sequence']))
            scene_info_index[scene_name] = scene_info
            logger.debug(f'Scene info {scene_info}.')

        return index, scene_info_index

    @staticmethod
    def _parse_labels(index: SceneIndex) -> Tuple[Dict[str, list], int]:
        """
        Loads all labels dictionary with format:
        {
            {scene_name}_{object_id}: {
                {frame_id}: [ymin, xmin, w, h]
            }
        }

        Args:
            index: Dataset File Index

        Returns:
            Labels dictionary
        """
        data = defaultdict(list)
        n_labels = 0

        for scene_name, scene_path in index.items():
            df = pd.read_csv(scene_path, header=None)
            df = df[df[7] == 1]  # Ignoring non-pedestrian objects

            df = df.iloc[:, :6]
            df.columns = ['frame_id', 'object_id', 'ymin', 'xmin', 'w', 'h']
            # df = df[['frame_id', 'object_id', 'xmin', 'ymin', 'w', 'h']]  # Reorder coordinates to xywh format
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
                        'image_path': None  # TODO
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
            'object_id': object_id
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

        # TODO: Images paths

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
            future_len: Number of unobserved data poitns
            label_type: Label Type
        """
        super(TorchMOTTrajectoryDataset, self).__init__()
        self._dataset = MOTDataset(
            path=path,
            history_len=history_len,
            future_len=future_len,
            label_type=label_type
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, _ = self._dataset[index]  # FIXME: Ignoring metadata for now
        bboxes_obs = torch.from_numpy(bboxes_obs)
        bboxes_unobs = torch.from_numpy(bboxes_unobs)
        ts_obs = torch.from_numpy(ts_obs)
        ts_unobs = torch.from_numpy(ts_unobs)

        return bboxes_obs, bboxes_unobs, ts_obs, ts_unobs

def run_test() -> None:
    from nodetracker.common.project import ASSETS_PATH
    configure_logging(logging.DEBUG)

    dataset_path = os.path.join(ASSETS_PATH, 'MOT20Labels', 'train')
    dataset = MOTDataset(dataset_path, history_len=4, future_len=4)

    print(f'Dataset size: {len(dataset)}')
    print(f'Sample example: {dataset[5]}')

    torch_dataset = TorchMOTTrajectoryDataset(dataset_path, history_len=4, future_len=4)

    print(f'Torch Dataset size: {len(torch_dataset)}')
    print(f'Torch Sample example shapes: {[x.shape for x in torch_dataset[5][:-1]]}')
    print(f'Torch Sample example: {torch_dataset[5]}')


if __name__ == '__main__':
    run_test()
