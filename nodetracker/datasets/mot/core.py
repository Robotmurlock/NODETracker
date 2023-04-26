"""
MOT Challenge Dataset support
"""
import configparser
import copy
import enum
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Any, Union, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from nodetracker.datasets.torch import TrajectoryDataset
from nodetracker.datasets.torch import run_dataset_test
from nodetracker.datasets.utils import split_trajectory_observed_unobserved
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
    category: str
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


class MOTDataset(TrajectoryDataset):
    """
    Parses MOT dataset in given format
    """
    def __init__(
        self,
        path: str,
        history_len: int,
        future_len: int,
        sequence_list: Optional[List[str]] = None,
        label_type: LabelType = LabelType.GROUND_TRUTH,
        **kwargs
    ) -> None:
        """
        Args:
            path: Path to dataset
            history_len: Number of observed data points
            future_len: Number of unobserved data points
            label_type: Label Type

        """
        super().__init__(history_len=history_len, future_len=future_len, sequence_list=sequence_list, **kwargs)

        self._path = path
        self._label_type = label_type

        self._scene_info_index = self._index_dataset(path, label_type, sequence_list)
        self._data_labels, self._n_labels, self._frame_to_data_index_lookup = self._parse_labels(self._scene_info_index)
        self._trajectory_index = self._create_trajectory_index(self._data_labels, self._history_len, self._future_len)

    @property
    def label_type(self) -> LabelType:
        """
        Returns: LabelType
        """
        return self._label_type

    @property
    def scenes(self) -> List[str]:
        return list(self._scene_info_index.keys())

    def parse_object_id(self, object_id: str) -> Tuple[str, str]:
        assert object_id in self._data_labels, f'Unknown object id "{object_id}".'
        scene_name, scene_object_id = object_id.split('_')
        return scene_name, scene_object_id

    def get_scene_object_ids(self, scene_name: str) -> List[str]:
        assert scene_name in self.scenes, f'Unknown scene "{scene_name}". Dataset scenes: {self.scenes}.'
        return [d for d in self._data_labels if d.startswith(scene_name)]

    def get_scene_number_of_object_ids(self, scene_name: str) -> int:
        return len(self.get_scene_object_ids(scene_name))

    def get_object_data_length(self, object_id: str) -> int:
        return len(self._data_labels[object_id])

    def get_object_data_label(self, object_id: str, index: int, relative_bbox_coords: bool = True) -> dict:
        data = copy.deepcopy(self._data_labels[object_id][index])

        if relative_bbox_coords:
            scene_name, _ = self.parse_object_id(object_id)
            scene_info = self._scene_info_index[scene_name]
            bbox = data['bbox']
            bbox = [
                bbox[0] / scene_info.imwidth,
                bbox[1] / scene_info.imheight,
                bbox[2] / scene_info.imwidth,
                bbox[3] / scene_info.imheight
            ]
            data['bbox'] = bbox

        return data

    def get_object_data_label_by_frame_index(self, object_id: str, frame_index: int, relative_bbox_coords: bool = True) -> Optional[dict]:
        index = self._frame_to_data_index_lookup[object_id].get(frame_index)
        if index is None:
            return index

        return self.get_object_data_label(object_id, index, relative_bbox_coords=relative_bbox_coords)

    def get_scene_info(self, scene_name: str) -> SceneInfo:
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
        scene_info = self._scene_info_index[scene_name]
        return self._get_image_path(scene_info, frame_id + 1)

    @staticmethod
    def _get_data_cache_path(path: str, data_name: str) -> str:
        """
        Get cache path for data object path.

        Args:
            path: Path

        Returns:
            Path where data object is or will be stored.
        """
        filename = Path(path).stem
        cache_filename = f'.{filename}.{data_name}.json'
        dirpath = str(Path(path).parent)
        return os.path.join(dirpath, cache_filename)

    @staticmethod
    def _index_dataset(
        path: str,
        label_type: LabelType,
        sequence_list: Optional[List[str]]
    ) -> SceneInfoIndex:
        """
        Index dataset content. Format: { {scene_name}: {scene_labels_path} }

        Args:
            path: Path to dataset
            label_type: Use ground truth bboxes or detections
            sequence_list: Filter scenes

        Returns:
            Index to scenes
        """
        scene_names = [file for file in os.listdir(path) if not file.startswith('.')]
        logger.debug(f'Found {len(scene_names)} scenes. Names: {scene_names}.')

        scene_info_index: SceneInfoIndex = {}

        for scene_name in scene_names:
            if sequence_list is not None and scene_name not in sequence_list:
                continue

            scene_directory = os.path.join(path, scene_name)
            scene_files = os.listdir(scene_directory)
            assert label_type.value in scene_files, f'Ground truth file "{label_type.value}" not found. ' \
                                                    f'Contents: {scene_files}'
            gt_path = os.path.join(scene_directory, label_type.value, f'{label_type.value}.txt')

            assert 'seqinfo.ini' in scene_files, f'Scene config file "seqinfo.ini" not found. Contents: {scene_files}'
            scene_info_path = os.path.join(scene_directory, 'seqinfo.ini')
            raw_info = configparser.ConfigParser()
            raw_info.read(scene_info_path)
            raw_info = dict(raw_info['Sequence'])
            raw_info['gt_path'] = gt_path
            raw_info['dirpath'] = scene_directory

            scene_info = SceneInfo(**raw_info, category='pedestrian')
            scene_info_index[scene_name] = scene_info
            logger.debug(f'Scene info {scene_info}.')

        return scene_info_index

    @staticmethod
    def _parse_labels(scene_infos: SceneInfoIndex) -> Tuple[Dict[str, list], int, Dict[str, Dict[int, int]]]:
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
        frame_to_data_index_lookup: Dict[str, Dict[int, int]] = defaultdict(dict)
        n_labels = 0

        for scene_name, scene_info in scene_infos.items():
            df = pd.read_csv(scene_info.gt_path, header=None)
            df = df[df[7] == 1]  # Ignoring non-pedestrian objects

            df = df.iloc[:, :6]
            df.columns = ['frame_id', 'object_id', 'ymin', 'xmin', 'w', 'h']  # format: yxwh
            df['object_global_id'] = \
                scene_name + '_' + df['object_id'].astype(str)  # object id is not unique over all scenes
            df = df.drop(columns='object_id', axis=1)
            df = df.sort_values(by=['object_global_id', 'frame_id'])
            n_labels += df.shape[0]

            object_groups = df.groupby('object_global_id')
            for object_global_id, df_grp in tqdm(object_groups, desc=f'Parsing {scene_name}', unit='pedestrian'):
                df_grp = df_grp.drop(columns='object_global_id', axis=1).set_index('frame_id')
                assert df_grp.index.max() - df_grp.index.min() + 1 == df_grp.index.shape[0], \
                    f'Object {object_global_id} has missing data points!'

                for frame_id, row in df_grp.iterrows():
                    data[object_global_id].append({
                        'frame_id': frame_id,
                        'bbox': row.values.tolist(),
                        'image_path': MOTDataset._get_image_path(scene_info, frame_id),
                        'occ': False,
                        'oov': False
                    })
                    frame_to_data_index_lookup[object_global_id][frame_id] = len(data[object_global_id]) - 1

        logger.debug(f'Parsed labels. Dataset size is {n_labels}.')
        data = dict(data)  # Disposing unwanted defaultdict side-effects
        return data, n_labels, frame_to_data_index_lookup

    @staticmethod
    def _create_trajectory_index(
        labels: Dict[str, list],
        history_len: int,
        future_len: int
    ) -> List[Tuple[str, int, int]]:
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

        for object_id, data in tqdm(labels.items(), desc='Creating trajectories', unit='object'):
            object_trajectory_len = len(data)
            for i in range(object_trajectory_len - trajectory_len + 1):
                traj_index.append((object_id, i, i + trajectory_len))

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
            'object_id': object_id,
            'frame_ids': frame_ids,
            'image_paths': [item['image_path'] for item in raw_traj]
        }

        # Bboxes
        bboxes = np.array([item['bbox'] for item in raw_traj], dtype=np.float32)
        # Normalize bbox coordinates
        bboxes[:, [0, 2]] /= scene_info.imwidth
        bboxes[:, [1, 3]] /= scene_info.imheight

        bboxes_obs, bboxes_unobs, frame_ts_obs, frame_ts_unobs = split_trajectory_observed_unobserved(frame_ids, bboxes, self._history_len)
        return bboxes_obs, bboxes_unobs, frame_ts_obs, frame_ts_unobs, metadata


if __name__ == '__main__':
    from nodetracker.common.project import ASSETS_PATH
    configure_logging(logging.DEBUG)

    dataset = MOTDataset(
        path=os.path.join(ASSETS_PATH, 'MOT20', 'train'),
        history_len=4,
        future_len=4
    )

    run_dataset_test(dataset)
