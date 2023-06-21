"""
LaSOT dataset. More information: https://paperswithcode.com/dataset/lasot
"""
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from tqdm import tqdm

from nodetracker.datasets.torch import TrajectoryDataset
from nodetracker.datasets.torch import run_dataset_test
from nodetracker.datasets.utils import split_trajectory_observed_unobserved
from nodetracker.utils import file_system
from nodetracker.utils.logging import configure_logging


@dataclass
class SequenceInfo:
    name: str
    category: str
    seqlength: int
    imheight: int
    imwidth: int
    text_label: str
    image_paths: List[str]
    bboxes: List[List[float]]
    time_points: List[int]
    occlusions: List[bool]
    out_of_views: List[bool]


FrameData = Tuple[List[int], bool, bool, str]  # coords, occlusion, out of view, image path


def parse_sequence(sequence_path: str) -> Tuple[List[FrameData], str]:
    """
    Parses sequence directory. Sequence data contains info for each frame:
    - bbox coordinates
    - is object occluded
    - is object out of view

    Sequence also contains text label.

    Args:
        sequence_path: Sequence path

    Returns:
        Parsed sequence with data:
        - Sequence data (bbox, occlusions, out of view)
        - text label
        - path to sequence images
    """
    occlusion_filepath = os.path.join(sequence_path, 'full_occlusion.txt')
    out_of_view_filepath = os.path.join(sequence_path, 'out_of_view.txt')
    nlp_filepath = os.path.join(sequence_path, 'nlp.txt')
    gt_filepath = os.path.join(sequence_path, 'groundtruth.txt')

    with open(occlusion_filepath, 'r', encoding='utf-8') as f:
        occlusion_raw = f.read()
        sep = ',' if ',' in occlusion_raw else '\n'  # LaSOT extension format is not consistent with old one
        occlusions = [bool(int(o.strip())) for o in occlusion_raw.strip().split(sep)]

    with open(out_of_view_filepath, 'r', encoding='utf-8') as f:
        oov_raw = f.read()
        sep = ',' if ',' in occlusion_raw else '\n'
        oov = [bool(int(o)) for o in oov_raw.strip().split(sep)]

    if os.path.exists(nlp_filepath):
        with open(nlp_filepath, 'r', encoding='utf-8') as f:
            text_label = f.read().strip()
    else:
        # LaSOT extension does not have `nlp.txt` files
        text_label = ''

    with open(gt_filepath, 'r', encoding='utf-8') as f:
        coord_lines = f.readlines()
        coords = [[int(v) for v in c.strip().split(',')] for c in coord_lines]

    image_dirpath = os.path.join(sequence_path, 'img')
    image_filenames = sorted(file_system.listdir(image_dirpath, regex_filter='(.*?)(jpg|png)'))
    image_paths = [os.path.join(image_dirpath, image_filename) for image_filename in image_filenames]

    assert len(occlusions) == len(oov) == len(coords), 'Failed to parse sequence!'
    data = list(zip(coords, occlusions, oov, image_paths))
    return data, text_label


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
        category_list: Optional[List[str]] = None,
        skip_occlusion: bool = False,
        skip_out_of_view: bool = False,
        **kwargs
    ):
        """

        Args:
            path: Dataset path
            history_len: Observed trajectory length
            future_len: Unobserved trajectory length
            sequence_list: Sequence list for dataset split
            skip_occlusion: Skip trajectories with occlusions
            skip_out_of_view: Skip trajectories that are out of view
        """
        super().__init__(history_len=history_len, future_len=future_len, sequence_list=sequence_list, **kwargs)

        self._sequence_index = self._create_dataset_index(
            path=path,
            sequence_list=sequence_list,
            category_list=category_list
        )
        self._trajectory_index = self._create_trajectory_index(
            sequence_index=self._sequence_index,
            history_len=history_len,
            future_len=future_len,
            skip_occlusion=skip_occlusion,
            skip_out_of_view=skip_out_of_view
        )

    # noinspection PyUnresolvedReferences
    @staticmethod
    def _create_dataset_index(
        path: str,
        sequence_list: Optional[List[str]],
        category_list: Optional[List[str]]
    ) -> SequenceInfoIndex:
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
            if category_list is not None and category not in category_list:
                continue

            if sequence_list is not None and sequence_name not in sequence_list:
                continue

            sequence_path = os.path.join(path, sequence_name)
            sequence_data, text_label = parse_sequence(sequence_path)
            coords, occlusions, oov, image_paths = zip(*sequence_data)

            # Load image to extract sequence image resolution
            image = cv2.imread(image_paths[0])
            assert image is not None, f'Failed to load image {image_paths[0]}!'

            h, w, _ = image.shape

            gt_path = os.path.join(sequence_path, 'groundtruth.txt')
            with open(gt_path, 'r', encoding='utf-8') as f:  # TODO: Why not use coords?
                lines = f.readlines()
                bboxes = [[int(v) for v in line.split(',')] for line in lines]
                bboxes = [[b[0] / w, b[1] / h, b[2] / w, b[3] / h] for b in bboxes]

            seqlength = len(image_paths)
            sequence_info = SequenceInfo(
                name=sequence_name,
                category=category,
                seqlength=seqlength,
                imheight=h,
                imwidth=w,
                text_label=text_label,
                image_paths=image_paths,
                bboxes=bboxes,
                time_points=list(range(seqlength)),
                occlusions=occlusions,
                out_of_views=oov
            )

            index[category][sequence_name] = sequence_info

        return dict(index)

    @staticmethod
    def _create_trajectory_index(
        sequence_index: SequenceInfoIndex,
        history_len: int,
        future_len: int,
        skip_occlusion: bool,
        skip_out_of_view: bool
    ) -> TrajectoryIndex:
        """
        Creates trajectory index (using indices).

        Args:
            sequence_index: Sequence info index
            history_len: Observed trajectory length
            future_len: Unobserved trajectory length
            skip_occlusion: Skip frames with object that is occluded
            skip_out_of_view: Skip frames with object that is out of view

        Returns:
            Trajectory index
        """
        trajectory_len = history_len + future_len
        traj_index: TrajectoryIndex = []

        pbar = tqdm(
            iterable=sequence_index.items(),
            total=len(sequence_index),
            unit='category',
            desc='Creating trajectory index'
        )
        for category, category_data in pbar:
            for sequence_name, sequence_info in category_data.items():
                traj_time_points = list(range(sequence_info.seqlength - trajectory_len + 1))

                for i in traj_time_points:
                    if skip_occlusion:
                        if any(sequence_info.occlusions[j] for j in range(i, i + trajectory_len)):
                            continue

                    if skip_out_of_view:
                        if any(sequence_info.out_of_views[j] for j in range(i, i + trajectory_len)):
                            continue

                    traj_index.append((category, sequence_name, i, i + trajectory_len))

        return traj_index

    @property
    def scenes(self) -> List[str]:
        return [sequence for category_sequences in self._sequence_index.values()
                for sequence in category_sequences.keys()]

    def get_scene_info(self, scene_name: str) -> Any:
        category = self.get_object_category(scene_name)
        return self._sequence_index[category][scene_name]

    def parse_object_id(self, object_id: str) -> Tuple[str, str]:
        category = self.get_object_category(object_id)
        assert category in self._sequence_index, f'Failed to find category "{category}"!'
        assert object_id in self._sequence_index[category], f'Failed to find sequence "{object_id}"!'

        return object_id, object_id

    def get_object_category(self, object_id: str) -> str:
        parts = object_id.split('-')
        assert len(parts) == 2, f'Failed to parse "{object_id}"!'
        return parts[0]

    def get_scene_object_ids(self, scene_name: str) -> List[str]:
        return [scene_name]  # Only one object per scene (equivalent to object_id) for SOT

    def get_object_data_length(self, object_id: str) -> int:
        category = self.get_object_category(object_id)
        sequence_info = self._sequence_index[category][object_id]
        return sequence_info.seqlength

    def get_object_data_label(self, object_id: str, index: int, relative_bbox_coords: bool = True) -> dict:
        # scene_name == object_id
        category = self.get_object_category(object_id)
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
        occlusion = sequence_info.occlusions[index]
        out_of_view = sequence_info.out_of_views[index]
        return {
            'frame_id': frame_id,
            'bbox': bbox,
            'image_path': image_path,
            'occ': occlusion,
            'oov': out_of_view,
            'category': category
        }

    def get_object_data_label_by_frame_index(
        self,
        object_id: str,
        frame_index: int,
        relative_bbox_coords: bool = True
    ) -> Optional[dict]:
        return self.get_object_data_label(object_id, frame_index, relative_bbox_coords=relative_bbox_coords)

    def get_scene_image_path(self, scene_name: str, frame_id: int) -> str:
        # scene_name == object_id
        category = self.get_object_category(scene_name)
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
            'category': category,
            'scene_name': sequence_name,
            'object_id': sequence_name,
            'frame_ids': frame_ids,
            'image_paths': image_paths
        }

        bboxes_obs, bboxes_unobs, frame_ts_obs, frame_ts_unobs = \
            split_trajectory_observed_unobserved(frame_ids, bboxes, self._history_len)
        return bboxes_obs, bboxes_unobs, frame_ts_obs, frame_ts_unobs, metadata


if __name__ == '__main__':
    from nodetracker.common.project import ASSETS_PATH
    configure_logging(logging.DEBUG)

    BAR = 80 * '=' + '\n\n'
    configuration = [
        ('None', False, False),
        ('Occlusions', True, False),
        ('Out of view', False, True),
        ('Occlusions and Out of view', True, True)
    ]

    first_iteration = True
    for desc, skip_occlusion_value, skip_out_of_view_value in configuration:
        if not first_iteration:
            print(BAR)
        first_iteration = False

        print(f'Testing dataset. Skip: {desc}')
        run_dataset_test(
            LaSOTDataset(
                path=os.path.join(ASSETS_PATH, 'LaSOT'),
                history_len=4,
                future_len=4,
                skip_occlusion=skip_occlusion_value,
                skip_out_of_view=skip_out_of_view_value
            )
        )
