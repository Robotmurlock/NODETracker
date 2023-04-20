import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from tqdm import tqdm

from nodetracker.datasets.torch import TrajectoryDataset
from nodetracker.datasets.torch import run_dataset_test
from nodetracker.datasets.utils import split_trajectory_observed_unobserved
from nodetracker.utils.logging import configure_logging


def listdir(path: str) -> List[str]:
    """
    Wrapper for `os.listdir` that ignore `.*` files (like `.DS_Store)

    Args:
        path: Directory path

    Returns:
        Listed files (not hidden)
    """
    return [p for p in os.listdir(path) if not p.startswith('.')]


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
    def __init__(
        self,
        path: str,
        history_len: int,
        future_len: int
    ):
        super().__init__(history_len=history_len, future_len=future_len)

        self._sequence_index = self._create_dataset_index(path)
        self._trajectory_index = self._create_trajectory_index(self._sequence_index, history_len, future_len)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def _create_dataset_index(path: str) -> SequenceInfoIndex:
        """
        Indexes all dataset metadata (loads everything except images) given the path

        Args:
            path: Path to the dataset

        Returns:
            SequenceIndex
        """
        index: SequenceInfoIndex = {}

        categories = listdir(path)
        for category in tqdm(categories, unit='category', desc='Indexing categories'):
            category_path = os.path.join(path, category)
            sequence_names = listdir(category_path)

            index[category]: Dict[str, SequenceInfo] = {}

            for sequence_name in sequence_names:
                sequence_path = os.path.join(category_path, sequence_name)
                sequence_images_path = os.path.join(sequence_path, 'img')
                image_filenames = sorted(listdir(sequence_images_path))
                image_paths = [os.path.join(sequence_images_path, filename) for filename in image_filenames]

                # Load image to extract sequence image resolution
                image = cv2.imread(image_paths[0])
                assert image is not None, f'Failed to load image {image_paths[0]}!'

                h, w, _ = image.shape

                gt_path = os.path.join(sequence_path, 'groundtruth.txt')
                with open(gt_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    bboxes = [[int(v) for v in line.split(',')] for line in lines]
                    bboxes = [[b[1] / w, b[0] / h, b[3] / w, b[2] / h] for b in bboxes]

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

        return index

    # noinspection PyMethodMayBeStatic
    def _create_trajectory_index(self, sequence_index: SequenceInfoIndex, history_len: int, future_len: int) -> TrajectoryIndex:
        """
        Creates trajectory index (using indices

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
                    traj_index.append((category, sequence_info.name, i, i + trajectory_len))

        return traj_index

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
        metadata = {
            'category': category,
            'name': sequence_name,
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
