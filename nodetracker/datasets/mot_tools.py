"""
Mot dataset utils.
"""
import argparse
import logging
import random
from typing import Tuple

import numpy as np
from tqdm import tqdm
from pathlib import Path

from nodetracker.datasets.mot import MOTDataset
from nodetracker.utils.logging import configure_logging

logger = logging.getLogger('MotTools')


def xyhw_add_noise(xmin: float, ymin: float, h: float, w: float, noise: float) -> Tuple[float, float, float, float]:
    """
    Add noise too coordinates.

    Args:
        xmin: up
        ymin: left
        h: height
        w: width
        noise: std

    Returns:
        xmin, ymin, h, w with added noise
    """
    values = []
    for val in [xmin, ymin, h, w]:
        val += np.random.normal(0, noise)
        val = max(0.0, min(val, 1.0))  # clip
        values.append(val)

    xmin, ymin, h, w = values
    return xmin, ymin, h, w

def create_tracker_sample(
    dataset: MOTDataset,
    scene_name: str,
    path: str,
    n_objects: int = 5,
    detections_noise: float = 1e-2,
    detections_skip_proba: float = 5e-2
) -> None:
    """
    Choose random `n_objects` ids and create tracker sample dataset from it.

    Args:
        dataset: Dataset which to use to create tracker sample.
        scene_name: Scene name.
        path: Path where to store tracker sample data.
        n_objects: Number of objects (tracklets) to use.
        detections_noise: Detections noise
        detections_skip_proba: Probability to skip some object detection in some frame.
    """
    object_ids = dataset.get_scene_object_ids(scene_name)
    total_object_ids = len(object_ids)
    logger.info(f'Found {total_object_ids} datapoints for scene "{scene_name}".')
    if total_object_ids < n_objects:
        logger.warning(f'Not enough objects. Sampling {total_object_ids} instead of {n_objects}.')
        n_objects = total_object_ids

    object_ids = random.sample(object_ids, n_objects)
    logger.info(f'Chosen objects: {object_ids}')

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as writer:
        header = 'scene_name,object_id,frame_id,image_path,xmin,ymin,h,w'
        writer.write(header)

        for object_id in tqdm(object_ids, unit='sample', desc='Creating tracker sample'):
            n_data_points = dataset.get_object_data_length(object_id)
            _, object_scene_id = dataset.parse_object_id(object_id)
            for index in range(n_data_points):
                r_skip = np.random.uniform()
                if r_skip <= detections_skip_proba:
                    continue

                data = dataset.get_object_data_label(object_id, index)
                frame_id = data['frame_id']
                image_path = data['image_path']
                xmin, ymin, h, w = data['bbox']
                xmin, ymin, h, w = xyhw_add_noise(xmin, ymin, h, w, noise=detections_noise)

                writer.write(f'\n{scene_name},{object_scene_id},{frame_id},{image_path},{xmin},{ymin},{h},{w}')


def parse_args() -> argparse.Namespace:
    """
    Returns:
        Parse tool arguments
    """
    parser = argparse.ArgumentParser(description='Create tracker sample.')
    parser.add_argument('--input-path', type=str, required=True, help='Dataset path.')
    parser.add_argument('--scene-name', type=str, required=True, help='Dataset scene name.')
    parser.add_argument('--output-path', type=str, required=True, help='Output path.')
    parser.add_argument('--objects', type=int, default=5, help='Number of objects in sample dataset.')
    parser.add_argument('--det-noise', type=float, default=1e-2, help='Detection coords noise.')
    parser.add_argument('--det-skip-proba', type=float, default=0.1, help='Detection skip frame probability')
    parser.add_argument('--seed', type=int, default=32, help='Seed')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    assert args.output_path.endswith('.csv'), f'Output path should have ".csv" extension. Got: "{args.output_path}".'
    random.seed(args.seed)
    dataset = MOTDataset(path=args.input_path, history_len=1, future_len=1, scene_filter=[args.scene_name])

    create_tracker_sample(
        dataset=dataset,
        scene_name=args.scene_name,
        path=args.output_path,
        n_objects=args.objects,
        detections_noise=args.det_noise,
        detections_skip_proba=args.det_skip_proba
    )


if __name__ == '__main__':
    configure_logging(logging.DEBUG)
    main(parse_args())
