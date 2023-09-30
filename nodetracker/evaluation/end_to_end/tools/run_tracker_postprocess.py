"""
Tracker inference postprocess.
"""
import logging
import os
from collections import Counter, defaultdict
from typing import Any, List

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets.factory import dataset_factory
from nodetracker.evaluation.end_to_end.config import TrackerGlobalConfig
from nodetracker.evaluation.end_to_end.tracker_utils import TrackerInferenceWriter, TrackerInferenceReader
from nodetracker.tracker import Tracklet
from nodetracker.utils import pipeline

logger = logging.getLogger('TrackerVizualization')


def element_distance_from_list(query: int, keys: List[int]) -> int:
    min_distance = None
    for key in keys:
        distance = abs(query - key)
        min_distance = distance if min_distance is None else min(distance, min_distance)

    return min_distance



@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='tracker_visualization', cls=TrackerGlobalConfig)
    cfg: TrackerGlobalConfig
    tracker_output = os.path.join(experiment_path, cfg.tracker.output_path, cfg.eval.split,
                                  cfg.tracker.object_detection.type, cfg.tracker.algorithm.name, )
    assert os.path.exists(tracker_output), f'Path "{tracker_output}" does not exist!'
    logger.info(f'Visualizing tracker inference on path "{tracker_output}".')

    tracker_all_output = os.path.join(tracker_output, 'all')
    tracker_active_output = os.path.join(tracker_output, 'active')
    tracker_postprocess_output = os.path.join(tracker_output, 'postprocess')

    dataset = dataset_factory(
        name=cfg.dataset.name,
        path=cfg.dataset.fullpath,
        history_len=1,  # Not relevant
        future_len=1,  # not relevant
        sequence_list=cfg.dataset.split_index[cfg.eval.split],
        additional_params=cfg.dataset.additional_params
    )

    scene_names = dataset.scenes
    for scene_name in tqdm(scene_names, desc='Postprocessing tracker', unit='scene'):
        scene_info = dataset.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        imheight = scene_info.imheight
        imwidth = scene_info.imwidth

        tracklet_presence_counter = Counter()  # Used to visualize new, appearing tracklets
        tracklet_frame_indices = defaultdict(list)
        with TrackerInferenceReader(tracker_active_output, scene_name, image_height=imheight, image_width=imwidth) as tracker_inf_reader:
            last_read = tracker_inf_reader.read()

            for index in tqdm(range(scene_length), desc=f'Counting occurrences "{scene_name}"', unit='frame'):
                if last_read is not None and index == last_read.frame_index:
                    for tracklet_id, bbox in last_read.objects.items():
                        tracklet_presence_counter[tracklet_id] += 1
                        tracklet_frame_indices[tracklet_id].append(index)

                    last_read = tracker_inf_reader.read()

        # Filter short tracklets (they have less than X active frames)
        tracklets_to_keep = {k for k, v in tracklet_presence_counter.items() if v >= 20}
        tracklet_frame_indices = dict(tracklet_frame_indices)

        with TrackerInferenceReader(tracker_all_output, scene_name, image_height=imheight, image_width=imwidth) as tracker_all_inf_reader, \
            TrackerInferenceWriter(tracker_postprocess_output, scene_name, image_height=imheight, image_width=imwidth) as tracker_inf_writer:

            last_all_read = tracker_all_inf_reader.read()

            for index in tqdm(range(scene_length), desc=f'Postprocessing "{scene_name}"', unit='frame'):
                if last_all_read is not None and index == last_all_read.frame_index:
                    for tracklet_id, bbox in last_all_read.objects.items():

                        if tracklet_id not in tracklets_to_keep:
                            # Marked as FP in postprocessing
                            continue

                        if element_distance_from_list(index, tracklet_frame_indices[tracklet_id]) > 1:
                            continue

                        tracklet = Tracklet(bbox=bbox, frame_index=index, _id=int(tracklet_id))
                        tracker_inf_writer.write(frame_index=index, tracklet=tracklet)

                    last_all_read = tracker_all_inf_reader.read()


if __name__ == '__main__':
    main()
