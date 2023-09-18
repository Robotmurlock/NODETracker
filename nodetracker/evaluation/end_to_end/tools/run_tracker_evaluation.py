"""
Tracker inference.
"""
import json
import logging
import os
from typing import List

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets.factory import dataset_factory
from nodetracker.evaluation.end_to_end.config import TrackerGlobalConfig
from nodetracker.evaluation.end_to_end.object_detection import object_detection_inference_factory, create_bbox_objects
from nodetracker.evaluation.end_to_end.tracker_utils import TrackerInferenceWriter
from nodetracker.tracker import tracker_factory, Tracklet
from nodetracker.utils import pipeline
from nodetracker.utils.lookup import LookupTable

logger = logging.getLogger('TrackerEvaluation')



@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='tracker_evaluation', cls=TrackerGlobalConfig)
    cfg: TrackerGlobalConfig
    tracker_output = os.path.join(experiment_path, cfg.tracker.output_path)
    logger.info(f'Saving tracker inference on path "{tracker_output}".')

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
    for scene_name in tqdm(scene_names, desc='Simulating tracker', unit='scene'):
        scene_length = dataset.get_scene_info(scene_name).seqlength

        with TrackerInferenceWriter(tracker_output, scene_name) as tracker_inf_writer:
            tracklets: List[Tracklet] = []
            for index in tqdm(range(scene_length), desc=f'Simulating "{scene_name}"', unit='frame'):
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
                    tracker_inf_writer.write(index, tracklet)


if __name__ == '__main__':
    main()
