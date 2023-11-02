"""
Tracker inference.
"""
import copy
import json
import logging
import os
import re
from dataclasses import asdict
from typing import List, Dict, Any

import hydra
import torch
import yaml
from omegaconf import DictConfig
from tqdm import tqdm

from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets import transforms
from nodetracker.datasets.factory import dataset_factory
from nodetracker.evaluation.end_to_end.config import TrackerGlobalConfig
from nodetracker.evaluation.end_to_end.object_detection import object_detection_inference_factory, create_bbox_objects
from nodetracker.evaluation.end_to_end.tracker_utils import TrackerInferenceWriter
from nodetracker.tracker import tracker_factory, Tracklet
from nodetracker.utils import pipeline
from nodetracker.utils.lookup import LookupTable
from tools.utils import create_inference_model

logger = logging.getLogger('TrackerEvaluation')


def populate_tracker_params(
    name: str,
    params: Dict[str, Any],
    cfg: TrackerGlobalConfig,
    experiment_path: str
) -> Dict[str, Any]:
    """
    Add additional tracker params (if needed)

    Args:
        name: Tracker algorithm name
        params: Tracker params
        cfg: Tracker inference/evaluation config
        experiment_path: Experiment path

    Returns:
        Populated tracker params
    """
    for key in ['filter_name', 'filter_params']:
        assert key in params, f'Expected `{key}` field for filter based tracker!'

    if not params['filter_name'] == 'akf':
        model = create_inference_model(cfg, experiment_path)
        transform_func = transforms.transform_factory(cfg.transform.name, cfg.transform.params)
        params = copy.deepcopy(params)
        params['filter_params']['model'] = model
        params['filter_params']['transform'] = transform_func

    return params


@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='tracker_evaluation', cls=TrackerGlobalConfig)
    cfg: TrackerGlobalConfig
    tracker_name = cfg.tracker.algorithm.name + (f'_{cfg.tracker.suffix}' if cfg.tracker.suffix is not None else '')
    tracker_output = os.path.join(experiment_path, cfg.tracker.output_path, cfg.eval.split,
                                  cfg.tracker.object_detection.type, tracker_name)
    assert not os.path.exists(tracker_output), f'Path "{tracker_output}" is already taken!'
    tracker_active_output = os.path.join(tracker_output, 'active')
    tracker_all_output = os.path.join(tracker_output, 'all')

    logger.info(f'Saving tracker inference on path "{tracker_output}".')

    with open(cfg.tracker.lookup_path, 'r', encoding='utf-8') as f:
        lookup = LookupTable.deserialize(json.load(f))

    additional_params = cfg.dataset.additional_params
    if cfg.dataset.name in ['DanceTrack', 'MOT20'] and cfg.eval.split == 'test':
        additional_params['test'] = True  # Skip labels parsing

    dataset = dataset_factory(
        name=cfg.dataset.name,
        path=cfg.dataset.fullpath,
        history_len=1,  # Not relevant
        future_len=1,  # not relevant
        sequence_list=cfg.dataset.split_index[cfg.eval.split],
        additional_params=additional_params
    )

    od_inference = object_detection_inference_factory(
        name=cfg.tracker.object_detection.type,
        params=cfg.tracker.object_detection.params,
        dataset=dataset,
        lookup=lookup
    )

    tracker_params = populate_tracker_params(
        name=cfg.tracker.algorithm.name,
        params=cfg.tracker.algorithm.params,
        cfg=cfg,
        experiment_path=experiment_path
    )

    scene_names = dataset.scenes
    scene_names = [scene_name for scene_name in scene_names if re.match(cfg.tracker.scene_pattern, scene_name)]
    for scene_name in tqdm(scene_names, desc='Simulating tracker', unit='scene'):
        scene_info = dataset.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        imheight = scene_info.imheight
        imwidth = scene_info.imwidth

        tracker = tracker_factory(
            name=cfg.tracker.algorithm.name,
            params=tracker_params
        )

        clip = cfg.dataset.name != 'MOT17'
        with TrackerInferenceWriter(tracker_active_output, scene_name, image_height=imheight, image_width=imwidth, clip=clip) as tracker_active_inf_writer, \
            TrackerInferenceWriter(tracker_all_output, scene_name, image_height=imheight, image_width=imwidth, clip=clip) as tracker_all_inf_writer:
            tracklets: List[Tracklet] = []
            for index in tqdm(range(scene_length), desc=f'Simulating "{scene_name}"', unit='frame'):
                # Perform OD inference
                inf_bboxes, inf_classes, inf_conf = od_inference.predict(
                    scene_name=scene_name,
                    frame_index=index
                )
                detection_bboxes = create_bbox_objects(inf_bboxes, inf_classes, inf_conf, clip=False)
                detection_bboxes = [bbox for bbox in detection_bboxes if bbox.aspect_ratio <= 1.6]

                # Perform tracking step
                active_tracklets, tracklets = tracker.track(
                    tracklets=tracklets,
                    detections=detection_bboxes,
                    frame_index=index + 1  # Counts from 1 instead of 0
                )

                # Save inference
                for tracklet in active_tracklets:
                    tracker_active_inf_writer.write(index, tracklet)

                for tracklet in tracklets:
                    tracker_all_inf_writer.write(index, tracklet)


    # Save tracker config
    tracker_config_path = os.path.join(tracker_output, 'config.yaml')
    with open(tracker_config_path, 'w', encoding='utf-8') as f:
        data = asdict(cfg.tracker)
        yaml.safe_dump(data, f)


if __name__ == '__main__':
    main()
