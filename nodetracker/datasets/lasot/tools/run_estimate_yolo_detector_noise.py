"""
YOLO model noise estimation.
"""
import argparse
import json
import logging
import re

import cv2
import numpy as np
import ultralytics
import yaml
from tqdm import tqdm

from nodetracker.datasets.lasot.core import LaSOTDataset
from nodetracker.evaluation.metrics.sot import iou
from nodetracker.utils.logging import configure_logging
from nodetracker.utils.lookup import LookupTable

logger = logging.getLogger('YOLOv8NoiseEstimation')


def parse_args() -> argparse.Namespace:
    """
    LaSOT to YOLOv8.

    Returns:
        Parsed configuration.
    """
    parser = argparse.ArgumentParser(description='Sequence analysis - visualization')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--model-path', type=str, required=True, help='Path where trained model is stored.')
    parser.add_argument('--lookup-path', type=str, required=True, help='Path where the YOLO class lookup is stored.')
    parser.add_argument('--split-index-path', type=str, required=True, help='Path where the split index is stored')
    parser.add_argument('--split', type=str, required=False, default='train', help='Split to perform')
    parser.add_argument('--min-conf', type=float, required=False, default=0.5, help='Minimum bbox confidence.')
    parser.add_argument('--min-iou', type=float, required=False, default=0.1, help='Minimum iou to accept prediction.')
    parser.add_argument('--scene-filter', type=str, required=True, help='Sequence filter - regex (not optional)')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    min_conf: float = args.min_conf
    min_iou: float = args.min_iou
    scene_filter: str = args.scene_filter

    dataset = LaSOTDataset(args.dataset_path, history_len=1, future_len=1)
    yolo = ultralytics.YOLO(args.model_path)
    with open(args.lookup_path, 'r', encoding='utf-8') as f:
        lookup = LookupTable.deserialize(json.load(f))
    with open(args.split_index_path, 'r', encoding='utf-8') as f:
        split_index = yaml.safe_load(f)

    diff_data = []


    scene_names = [scene_name for scene_name in dataset.scenes if re.match(scene_filter, scene_name)]
    scene_names = [scene_name for scene_name in scene_names if scene_name in split_index[args.split]]
    logger.info(f'Total number of scenes to process: {len(scene_names)}')
    for scene_name in tqdm(scene_names, unit='scene'):
        for object_id in dataset.get_scene_object_ids(scene_name):
            length = dataset.get_object_data_length(object_id)
            for i in range(length):
                data = dataset.get_object_data_label_by_frame_index(object_id, i)
                img_path = data['image_path']
                gt_bbox = np.array(data['bbox'])

                # noinspection PyUnresolvedReferences
                img = cv2.imread(img_path)
                assert img is not None, f'Failed to load image "{img_path}"!'

                prediction_raw = yolo.predict(
                    source=img,
                    conf=min_conf,
                    classes=lookup.lookup(data['category']),
                    verbose=False
                )[0]

                h, w, _ = img.shape
                bboxes = prediction_raw.boxes.xyxy.detach().cpu().numpy()
                bboxes[:, [0, 2]] /= w
                bboxes[:, [1, 3]] /= h
                bboxes[:, 2:] -= bboxes[:, :2]  # xyxy to xywh
                n_bboxes = bboxes.shape[0]

                best_fitting_bbox, best_fitting_iou = None, None
                for bbox_index in range(n_bboxes):
                    pred_bbox = bboxes[bbox_index]
                    iou_score = float(iou(pred_bbox, gt_bbox))
                    if iou_score < min_iou:
                        continue

                    if best_fitting_iou is None or iou_score > best_fitting_iou:
                        best_fitting_bbox = pred_bbox
                        best_fitting_iou = iou_score

                if best_fitting_bbox is None:
                    continue

                bbox_diff = gt_bbox - best_fitting_bbox
                # Converting to relative coordinates
                bbox_diff[:2] /= gt_bbox[2:]
                bbox_diff[2:] /= gt_bbox[2:]

                diff_data.append(bbox_diff)

    diff_data = np.stack(diff_data)
    diff_std = diff_data.std()
    logger.info(f'Estimated detector std: {diff_std:.4f}')


if __name__ == '__main__':
    configure_logging(logging.INFO)
    main(parse_args())
