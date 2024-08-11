"""
Analysis of lost tracker refind.
"""
import json
import logging
import os
import re
from collections import Counter
from typing import List

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets.factory import dataset_factory
from nodetracker.evaluation.end_to_end.config import TrackerGlobalConfig
from nodetracker.evaluation.end_to_end.tracker_utils import TrackerInferenceReader
from nodetracker.library.cv.bbox import PredBBox, BBox
from nodetracker.tracker.matching import HungarianAlgorithmIOU
from nodetracker.utils import pipeline

logger = logging.getLogger('TrackerAnalyzeRefind')


@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='tracker_analyze_refind', cls=TrackerGlobalConfig)
    cfg: TrackerGlobalConfig

    tracker_name = cfg.tracker.algorithm.name + (f'_{cfg.tracker.suffix}' if cfg.tracker.suffix is not None else '')
    tracker_output = os.path.join(experiment_path, cfg.tracker.output_path, cfg.eval.split,
                                  cfg.tracker.object_detection.type, tracker_name)
    tracker_output_active = os.path.join(tracker_output, 'active')
    assert os.path.exists(tracker_output_active), f'Path "{tracker_output_active}" does not exist!'
    assert cfg.eval.split != 'test', 'Analysis can\'t be performed on test set!'

    dataset = dataset_factory(
        name=cfg.dataset.name,
        path=cfg.dataset.fullpath,
        history_len=1,  # Not relevant
        future_len=1,  # not relevant
        sequence_list=cfg.dataset.split_index[cfg.eval.split],
        additional_params=cfg.dataset.additional_params
    )

    matcher = HungarianAlgorithmIOU(match_threshold=0.5)

    scene_names = dataset.scenes
    scene_names = [scene_name for scene_name in scene_names if re.match(cfg.tracker.scene_pattern, scene_name)]

    # Stats
    BIN_SIZE = 16
    LOST_THRESHOLD = 30
    MIN_OCCLUSION_LENGTH = 5
    tp_refind, fp_refind, any_refind = Counter(), Counter(), Counter()
    fn_total = 0

    for scene_name in tqdm(scene_names, desc='Analyzing refind on scenes', unit='scene'):
        scene_info = dataset.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        imheight = scene_info.imheight
        imwidth = scene_info.imwidth
        object_ids = dataset.get_scene_object_ids(scene_name)

        # State
        pred_last_frame_index = {}
        gt_last_frame_index = {}
        pred_id_to_gt_id = {}
        gt_id_to_pred_id = {}

        with TrackerInferenceReader(tracker_output_active, scene_name, image_height=imheight, image_width=imwidth) as tracker_inf_reader:
            last_read = tracker_inf_reader.read()

            for index in tqdm(range(scene_length), desc=f'Analyzing refind on "{scene_name}"', unit='frame'):
                if last_read is not None and index == last_read.frame_index:
                    # Form tracklet bboxes
                    tracklet_bboxes: List[PredBBox] = []
                    tracklet_ids: List[str] = []
                    for tracklet_id, bbox in last_read.objects.items():
                        tracklet_bboxes.append(bbox)
                        tracklet_ids.append(tracklet_id)

                    # Form GT tracklet bboxes
                    gt_tracklet_bboxes: List[PredBBox] = []
                    gt_tracklet_ids: List[str] = []
                    for object_id in object_ids:
                        data = dataset.get_object_data_label_by_frame_index(object_id, index)
                        if data is None:
                            continue

                        bbox = PredBBox.create(
                            bbox=BBox.from_yxwh(*data['bbox']),
                            label=TrackerInferenceReader.BBOX_LABEL,
                            conf=1.0
                        )

                        gt_tracklet_bboxes.append(bbox)
                        gt_tracklet_ids.append(object_id)

                    # Match preds with GT
                    matches, _, _ = matcher(tracklet_bboxes, gt_tracklet_bboxes)
                    for pred_i, gt_i in matches:
                        pred_id = tracklet_ids[pred_i]
                        gt_id = gt_tracklet_ids[gt_i]

                        # Check if there is a refind
                        if pred_id in pred_last_frame_index and index - pred_last_frame_index[pred_id] >= MIN_OCCLUSION_LENGTH \
                                and pred_id in pred_id_to_gt_id:
                            last_gt_id = pred_id_to_gt_id[pred_id]
                            time_diff = index - pred_last_frame_index[pred_id]
                            group_id = (time_diff - 1) // BIN_SIZE
                            group_name = f'{group_id * BIN_SIZE + 1}-{(group_id + 1) * BIN_SIZE}'

                            any_refind[group_name] += 1

                            if gt_id == last_gt_id:
                                tp_refind[group_name] += 1
                            else:
                                fp_refind[group_name] += 1

                        pred_id_to_gt_id[pred_id] = gt_id
                        gt_id_to_pred_id[gt_i] = pred_id
                        gt_last_frame_index[gt_i] = index

                    # Update state
                    gt_ids_to_delete: List[int] = []
                    for gt_id, last_frame_index in gt_last_frame_index.items():
                        if index - last_frame_index >= LOST_THRESHOLD:
                            fn_total += 1
                            gt_ids_to_delete.append(gt_id)

                    for gt_id in gt_ids_to_delete:
                        gt_last_frame_index.pop(gt_id)
                        gt_id_to_pred_id.pop(gt_id)

                    for tracklet_id in tracklet_ids:
                        pred_last_frame_index[tracklet_id] = index

                    last_read = tracker_inf_reader.read()



    tp_refind, fp_refind, any_refind = [dict(v) for v in [tp_refind, fp_refind, any_refind]]  # Convert to normal dict
    stats = {
        'total': {
            'matches': sum(any_refind.values()),
            'TP': sum(tp_refind.values()),
            'FP': sum(fp_refind.values()),
            'FN': fn_total
        },
        'distribution': {
            'matches': dict(sorted(list(any_refind.items()), key=lambda x: int(x[0].split('-')[0]))),
            'TP': dict(sorted(list(tp_refind.items()), key=lambda x: int(x[0].split('-')[0]))),
            'FP': dict(sorted(list(fp_refind.items()), key=lambda x: int(x[0].split('-')[0])))
        }
    }
    logger.info(f'Summary:\n{json.dumps(stats, indent=2)}')


if __name__ == '__main__':
    main()
