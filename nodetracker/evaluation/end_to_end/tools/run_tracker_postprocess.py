"""
Tracker inference postprocess.
"""
import logging
import os
import re
from collections import Counter, defaultdict
from typing import List, Dict

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets.factory import dataset_factory
from nodetracker.evaluation.end_to_end.config import TrackerGlobalConfig
from nodetracker.evaluation.end_to_end.tracker_utils import TrackerInferenceWriter, TrackerInferenceReader
from nodetracker.library.cv.bbox import PredBBox, Point
from nodetracker.tracker import Tracklet
from nodetracker.utils import pipeline

logger = logging.getLogger('TrackerVizualization')


INF = 999_999


def element_distance_from_list(query: int, keys: List[int]) -> int:
    min_distance = None
    for key in keys:
        distance = abs(query - key)
        min_distance = distance if min_distance is None else min(distance, min_distance)

    return min_distance


def find_closest_prev_element(query: int, keys: List[int]) -> int:
    value = None
    for key in keys:
        if key > query:
            break
        value = key

    assert value is not None
    return value


def find_closest_next_element(query: int, keys: List[int]) -> int:
    value = None
    for key in keys:
        if key < query:
            continue
        value = key
        break

    assert value is not None
    return value


def interpolate_bbox(start_index: int, start_bbox: PredBBox, end_index: int, end_bbox: PredBBox, index: int) -> PredBBox:
    """
    Perform linear interpolation between two bounding boxes at a specific index.

    Args:
        start_index: The frame index of the starting bounding box.
        start_bbox: The starting bounding box.
        end_index: The frame index of the ending bounding box.
        end_bbox: The ending bounding box.
        index: The target frame index for interpolation.

    Returns:
        Interpolated bounding box at the target index.
    """
    assert start_index < index < end_index, "The target index must be between start_index and end_index"

    # Calculate alpha based on the target index
    alpha = (index - start_index) / (end_index - start_index)

    # Interpolate upper left corner
    interpolated_upper_left = Point(
        start_bbox.upper_left.x + alpha * (end_bbox.upper_left.x - start_bbox.upper_left.x),
        start_bbox.upper_left.y + alpha * (end_bbox.upper_left.y - start_bbox.upper_left.y)
    )

    # Interpolate bottom right corner
    interpolated_bottom_right = Point(
        start_bbox.bottom_right.x + alpha * (end_bbox.bottom_right.x - start_bbox.bottom_right.x),
        start_bbox.bottom_right.y + alpha * (end_bbox.bottom_right.y - start_bbox.bottom_right.y)
    )

    # Interpolate label and confidence (if available)
    interpolated_label = start_bbox.label
    interpolated_conf = None
    if start_bbox.conf is not None and end_bbox.conf is not None:
        interpolated_conf = start_bbox.conf + alpha * (end_bbox.conf - start_bbox.conf)

    interpolated_bbox = PredBBox(
        upper_left=interpolated_upper_left,
        bottom_right=interpolated_bottom_right,
        label=interpolated_label,
        conf=interpolated_conf
    )

    return interpolated_bbox



@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='tracker_postprocess', cls=TrackerGlobalConfig)
    cfg: TrackerGlobalConfig
    tracker_name = cfg.tracker.algorithm.name + (f'_{cfg.tracker.suffix}' if cfg.tracker.suffix is not None else '')
    tracker_output = os.path.join(experiment_path, cfg.tracker.output_path, cfg.eval.split,
                                  cfg.tracker.object_detection.type, tracker_name)
    assert os.path.exists(tracker_output), f'Path "{tracker_output}" does not exist!'
    logger.info(f'Postprocessing tracker inference on path "{tracker_output}".')

    tracker_all_output = os.path.join(tracker_output, 'all')
    tracker_active_output = os.path.join(tracker_output, 'active')
    tracker_postprocess_output = os.path.join(tracker_output, 'postprocess')

    additional_params = cfg.dataset.additional_params
    if cfg.dataset.name in ['DanceTrack', 'MOT20', 'SportsMOT'] and cfg.eval.split == 'test':
        additional_params['test'] = True  # Skip labels parsing

    dataset = dataset_factory(
        name=cfg.dataset.name,
        path=cfg.dataset.fullpath,
        history_len=1,  # Not relevant
        future_len=1,  # not relevant
        sequence_list=cfg.dataset.split_index[cfg.eval.split],
        additional_params=additional_params
    )

    scene_names = dataset.scenes
    scene_names = [scene_name for scene_name in scene_names if re.match(cfg.tracker.scene_pattern, scene_name)]
    for scene_name in tqdm(scene_names, desc='Postprocessing tracker', unit='scene'):
        scene_info = dataset.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        imheight = scene_info.imheight
        imwidth = scene_info.imwidth

        tracklet_presence_counter = Counter()  # Used to visualize new, appearing tracklets
        tracklet_frame_bboxes: Dict[str, Dict[int, PredBBox]] = defaultdict(dict)
        with TrackerInferenceReader(tracker_active_output, scene_name, image_height=imheight, image_width=imwidth) as tracker_inf_reader:
            last_read = tracker_inf_reader.read()

            for index in tqdm(range(scene_length), desc=f'Counting occurrences "{scene_name}"', unit='frame'):
                if last_read is not None and index == last_read.frame_index:
                    for tracklet_id, bbox in last_read.objects.items():
                        tracklet_presence_counter[tracklet_id] += 1
                        tracklet_frame_bboxes[tracklet_id][index] = bbox

                    last_read = tracker_inf_reader.read()

        # (1) Filter short tracklets (they have less than X active frames)
        tracklets_to_keep = {k for k, v in tracklet_presence_counter.items() if v >= cfg.tracker.postprocess.min_tracklet_length}
        tracklet_frame_bboxes = dict(tracklet_frame_bboxes)

        clip = cfg.dataset.name != 'MOT17'
        with TrackerInferenceReader(tracker_all_output, scene_name, image_height=imheight, image_width=imwidth) as tracker_all_inf_reader, \
            TrackerInferenceWriter(tracker_postprocess_output, scene_name, image_height=imheight, image_width=imwidth, clip=clip) as tracker_inf_writer:

            last_all_read = tracker_all_inf_reader.read()

            for index in tqdm(range(scene_length), desc=f'Postprocessing "{scene_name}"', unit='frame'):
                if last_all_read is not None and index == last_all_read.frame_index:
                    for tracklet_id, bbox in last_all_read.objects.items():

                        if tracklet_id not in tracklets_to_keep:
                            # Marked as FP in postprocessing
                            continue

                        tracklet_indices = list(tracklet_frame_bboxes[tracklet_id].keys())
                        keep = False
                        if index in tracklet_indices:
                            keep = True

                        # (2) Linear interpolation
                        if index not in tracklet_indices and min(tracklet_indices) <= index <= max(tracklet_indices) \
                                and tracklet_presence_counter[tracklet_id] >= cfg.tracker.postprocess.linear_interpolation_min_tracklet_length:
                            prev_index = find_closest_prev_element(index, tracklet_indices)
                            next_index = find_closest_next_element(index, tracklet_indices)
                            if next_index - prev_index > cfg.tracker.postprocess.linear_interpolation_threshold:
                                continue

                            bbox = interpolate_bbox(
                                start_index=prev_index,
                                start_bbox=tracklet_frame_bboxes[tracklet_id][prev_index],
                                end_index=next_index,
                                end_bbox=tracklet_frame_bboxes[tracklet_id][next_index],
                                index=index
                            )
                            keep = True

                        # (3) Add trajectory initialization
                        start_index = min(tracklet_indices)
                        if index < start_index and start_index - index <= cfg.tracker.postprocess.init_threshold:
                            keep = True

                        if keep:
                            tracklet = Tracklet(bbox=bbox, frame_index=index, _id=int(tracklet_id))
                            tracker_inf_writer.write(frame_index=index, tracklet=tracklet)

                    last_all_read = tracker_all_inf_reader.read()


if __name__ == '__main__':
    main()
