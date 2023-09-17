"""
Tracker inference.
"""
import json
import logging
import os
from pathlib import Path
from typing import List
from typing import TextIO

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets.factory import dataset_factory
from nodetracker.evaluation.end_to_end.config import TrackerGlobalConfig
from nodetracker.evaluation.end_to_end.object_detection import object_detection_inference_factory, create_bbox_objects
from nodetracker.tracker import tracker_factory, Tracklet
from nodetracker.utils import pipeline
from nodetracker.utils.lookup import LookupTable

logger = logging.getLogger('TrackerEvaluation')


class TrackerInferenceWriter:
    """
    Writes tracker inference in MOT format. Can be used afterward for evaluation.
    (RAII)
    """
    def __init__(self, output_path: str, scene_name: str):
        """
        Args:
            output_path: Tracker inference output directory path
            scene_name: Scene name
        """
        self._output_path = output_path
        self._scene_name = scene_name
        self._scene_output_path = os.path.join(output_path, f'{scene_name}.txt')

        # State
        self._writer = None

    def open(self) -> None:
        """
        Opens writer.
        """
        Path(self._output_path).mkdir(parents=True, exist_ok=True)
        self._writer: TextIO = open(self._scene_output_path, 'w', encoding='utf-8')

    def close(self) -> None:
        """
        Closes writer.
        """
        self._writer.close()

    def write(self, tracklet: Tracklet) -> None:
        """
        Writes info about one tracker tracklet.
        One tracklet - one row.

        Args:
            tracklet: Tracklet
        """
        frame_id, bbox = tracklet.latest
        cells = [
            str(frame_id), str(tracklet.id),
            str(bbox.upper_left.y), str(bbox.upper_left.x),
            str(bbox.width), str(bbox.height),
            str(bbox.conf),
            '-1', '-1', '-1'
        ]

        row = ','.join(cells)
        self._writer.write(f'{row}\n')

    def __enter__(self) -> 'TrackerInferenceWriter':
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='tracker_evaluation', cls=TrackerGlobalConfig)
    cfg: TrackerGlobalConfig

    with open(cfg.tracker.lookup_path, 'r', encoding='utf-8') as f:
        lookup = LookupTable.deserialize(json.load(f))

    dataset = dataset_factory(
        name=cfg.dataset.name,
        path=cfg.dataset.fullpath,
        history_len=1,  # Not relevant
        future_len=1,  # not relevant
        sequence_list=cfg.dataset.split_index[cfg.eval.split],
        additional_params=cfg.dataset.additional_params
    )

    od_inference = object_detection_inference_factory(
        name=cfg.tracker.object_detection.type,
        params=cfg.tracker.object_detection.params,
        dataset=dataset,
        lookup=lookup
    )

    tracker = tracker_factory(
        name=cfg.tracker.algorithm.name,
        params=cfg.tracker.algorithm.params
    )

    scene_names = dataset.scenes
    for scene_name in tqdm(scene_names, desc='Evaluating tracker', unit='scene'):
        scene_length = dataset.get_scene_info(scene_name).seqlength

        with TrackerInferenceWriter(cfg.tracker.output_path, scene_name) as tracker_inf_writer:
            tracklets: List[Tracklet] = []
            for index in tqdm(range(scene_length), desc=f'Evaluating "{scene_name}"', unit='frame'):
                # Perform OD inference
                inf_bboxes, inf_classes, inf_conf = od_inference.predict(
                    scene_name=scene_name,
                    frame_index=index
                )
                detection_bboxes = create_bbox_objects(inf_bboxes, inf_classes, inf_conf)

                # Perform tracking step
                new_tracklets, updated_tracklets, deleted_tracklets = tracker.track(
                    tracklets=tracklets,
                    detections=detection_bboxes,
                    frame_index=index + 1  # Counts from 1 instead of 0
                )
                tracklets = new_tracklets + updated_tracklets

                # Save inference
                for tracklet in tracklets:
                    tracker_inf_writer.write(tracklet)


if __name__ == '__main__':
    main()
