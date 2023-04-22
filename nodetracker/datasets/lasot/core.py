"""
LaSOT dataset. More information: https://paperswithcode.com/dataset/lasot
"""
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from nodetracker.utils import file_system
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from nodetracker.datasets.torch import TrajectoryDataset
from nodetracker.datasets.torch import run_dataset_test
from nodetracker.datasets.utils import split_trajectory_observed_unobserved
from nodetracker.utils.logging import configure_logging


@dataclass
class SequenceInfo:
    name: str
    category: str
    length: int
    imheight: int
    imwidth: int
    image_paths: List[str]
    bboxes: List[List[float]]
    time_points: List[int]


SequenceInfoIndex = Dict[str, Dict[str, SequenceInfo]]  # Category -> (SequenceName -> SequenceInfo)
TrajectoryIndex = List[Tuple[str, str, int, int]]

class LaSOTDataset(TrajectoryDataset):
    """
    Indexes and loads LaSOT data.
    """
    def __init__(
        self,
        path: str,
        history_len: int,
        future_len: int,
        sequence_list: Optional[List[str]] = None,
        **kwargs
    ):
        """

        Args:
            path: Dataset path
            history_len: Observed trajectory length
            future_len: Unobserved trajectory length
            sequence_list: Sequence list for dataset split
        """
        super().__init__(history_len=history_len, future_len=future_len, sequence_list=sequence_list, **kwargs)

        self._sequence_index = self._create_dataset_index(path, sequence_list)
        self._trajectory_index = self._create_trajectory_index(self._sequence_index, history_len, future_len)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def _create_dataset_index(path: str, sequence_list: Optional[Dict[str, List[str]]]) -> SequenceInfoIndex:
        """
        Indexes all dataset metadata (loads everything except images) given the path

        Args:
            path: Path to the dataset
            sequence_list: Sequence list for dataset split

        Returns:
            SequenceIndex
        """
        index: SequenceInfoIndex = defaultdict(dict)

        sequences = file_system.listdir(path)
        for sequence_name in tqdm(sequences, unit='sequences', desc='Indexing sequences'):
            category, _ = sequence_name.split('-')
            if sequence_list is not None and sequence_name not in sequence_list:
                continue

            sequence_path = os.path.join(path, sequence_name)
            sequence_images_path = os.path.join(sequence_path, 'img')
            image_filenames = sorted(file_system.listdir(sequence_images_path, regex_filter='(.*?)(jpg|png)'))
            image_paths = [os.path.join(sequence_images_path, filename) for filename in image_filenames]

            # Load image to extract sequence image resolution
            image = cv2.imread(image_paths[0])
            assert image is not None, f'Failed to load image {image_paths[0]}!'

            h, w, _ = image.shape

            gt_path = os.path.join(sequence_path, 'groundtruth.txt')
            with open(gt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                bboxes = [[int(v) for v in line.split(',')] for line in lines]
                bboxes = [[b[0] / w, b[1] / h, b[2] / w, b[3] / h] for b in bboxes]

            seqlength = len(image_paths)
            sequence_info = SequenceInfo(
                name=sequence_name,
                category=category,
                length=seqlength,
                imheight=h,
                imwidth=w,
                image_paths=image_paths,
                bboxes=bboxes,
                time_points=list(range(seqlength))
            )

            index[category][sequence_name] = sequence_info

        return dict(index)

    # noinspection PyMethodMayBeStatic
    def _create_trajectory_index(self, sequence_index: SequenceInfoIndex, history_len: int, future_len: int) -> TrajectoryIndex:
        """
        Creates trajectory index (using indices).

        Args:
            sequence_index: Sequence info index
            history_len: Observed trajectory length
            future_len: Unobserved trajectory length

        Returns:

        """
        trajectory_len = history_len + future_len
        traj_index: TrajectoryIndex = []

        for category, category_data in tqdm(sequence_index.items(), total=len(sequence_index), unit='category', desc='Creating trajectory index'):
            for sequence_name, sequence_info in category_data.items():
                for i in range(sequence_info.length - trajectory_len + 1):
                    traj_index.append((category, sequence_name, i, i + trajectory_len))

        return traj_index

    @property
    def scenes(self) -> List[str]:
        return [sequence for category_sequences in self._sequence_index.values() for sequence in category_sequences.keys()]

    @staticmethod
    def _get_sequence_category(sequence_id: str) -> str:
        """
        Extracts category from `sequence_id`.

        Args:
            sequence_id: Sequence id

        Returns:
            Sequence category
        """
        parts = sequence_id.split('-')
        assert len(parts) == 2, f'Failed to parse "{sequence_id}"!'
        return parts[0]

    def parse_object_id(self, object_id: str) -> Tuple[str, str]:
        category = self._get_sequence_category(object_id)
        assert category in self._sequence_index, f'Failed to find category "{category}"!'
        assert object_id in self._sequence_index[category], f'Failed to find sequence "{object_id}"!'

        return object_id, object_id

    def get_scene_object_ids(self, scene_name: str) -> List[str]:
        return [scene_name]  # Only one object per scene (equivalent to object_id) for SOT

    def get_object_data_length(self, object_id: str) -> int:
        category = self._get_sequence_category(object_id)
        sequence_info = self._sequence_index[category][object_id]
        return sequence_info.length

    def get_object_data_label(self, object_id: str, index: int, relative_bbox_coords: bool = True) -> dict:
        # scene_name == object_id
        category = self._get_sequence_category(object_id)
        sequence_info = self._sequence_index[category][object_id]

        bbox = sequence_info.bboxes[index]
        if not relative_bbox_coords:
            bbox = [
                int(bbox[0] * sequence_info.imwidth),
                int(bbox[1] * sequence_info.imheight),
                int(bbox[2] * sequence_info.imwidth),
                int(bbox[3] * sequence_info.imheight)
            ]

        frame_id = index
        image_path = sequence_info.image_paths[index]
        return {
            'frame_id': frame_id,
            'bbox': bbox,
            'image_path': image_path
        }

    def get_scene_image_path(self, scene_name: str, frame_id: int) -> str:
        # scene_name == object_id
        category = self._get_sequence_category(scene_name)
        sequence_info = self._sequence_index[category][scene_name]
        return sequence_info.image_paths[frame_id]


    def __len__(self) -> int:
        return len(self._trajectory_index)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        category, sequence_name, traj_start, traj_end = self._trajectory_index[index]
        sequence_info = self._sequence_index[category][sequence_name]

        # Extract sequence data
        frame_ids = sequence_info.time_points[traj_start:traj_end]
        image_paths = sequence_info.image_paths[traj_start:traj_end]
        bboxes = np.array(sequence_info.bboxes[traj_start:traj_end], dtype=np.float32)

        # Metadata
        # scene == object for SOT but data is duplicated for MOT compatability
        metadata = {
            'scene_name': sequence_name,
            'object_id': sequence_name,
            'frame_ids': frame_ids,
            'image_paths': image_paths
        }

        bboxes_obs, bboxes_unobs, frame_ts_obs, frame_ts_unobs = split_trajectory_observed_unobserved(frame_ids, bboxes, self._history_len)
        return bboxes_obs, bboxes_unobs, frame_ts_obs, frame_ts_unobs, metadata


if __name__ == '__main__':
    from nodetracker.common.project import ASSETS_PATH
    configure_logging(logging.DEBUG)

    dataset = LaSOTDataset(
        path=os.path.join(ASSETS_PATH, 'LaSOT', 'train'),
        history_len=4,
        future_len=4
    )

    run_dataset_test(dataset)
