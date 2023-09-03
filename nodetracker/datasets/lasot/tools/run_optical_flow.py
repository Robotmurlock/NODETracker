"""
Visualize optical flow (motion) on LaSOT dataset
"""
import argparse
import logging
import os
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from nodetracker.common.project import OUTPUTS_PATH
from nodetracker.datasets.lasot.core import LaSOTDataset
from nodetracker.library.cv.video_writer import MP4Writer
from nodetracker.utils.logging import configure_logging

logger = logging.getLogger('OpticalFlowVisualization')


def parse_args() -> argparse.Namespace:
    """
    LaSOT to YOLOv8.

    Returns:
        Parsed configuration.
    """
    parser = argparse.ArgumentParser(description='Optical FLow Visualization')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--output-path', type=str, required=False, default=OUTPUTS_PATH,
                        help='Path where video outputs are stored.')
    parser.add_argument('--scene-filter', type=str, required=True, help='Sequence filter - regex (not optional)')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    scene_filter: str = args.scene_filter
    dataset = LaSOTDataset(args.dataset_path, history_len=1, future_len=1)

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    scene_names = [scene_name for scene_name in dataset.scenes if re.match(scene_filter, scene_name)]
    for scene_name in scene_names:
        video_path = os.path.join(args.output_path, f'{scene_name}.mp4')
        with MP4Writer(video_path, fps=30) as writer:
            for object_id in dataset.get_scene_object_ids(scene_name):
                length = dataset.get_object_data_length(object_id)
                for i in tqdm(range(length - 1)):
                    # Load current image
                    curr_data = dataset.get_object_data_label_by_frame_index(object_id, i)
                    curr_img_path = curr_data['image_path']
                    curr_img = cv2.imread(curr_img_path)
                    curr_img = cv2.resize(curr_img, (224, 224), interpolation=cv2.INTER_NEAREST)
                    curr_gray_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
                    assert curr_img is not None

                    # Load next image
                    next_data = dataset.get_object_data_label_by_frame_index(object_id, i + 1)
                    next_img_path = next_data['image_path']
                    next_img = cv2.imread(next_img_path)
                    next_img = cv2.resize(next_img, (224, 224), interpolation=cv2.INTER_NEAREST)
                    next_gray_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
                    assert next_img is not None

                    flow = cv2.calcOpticalFlowFarneback(curr_gray_img, next_gray_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv = np.zeros_like(curr_img)
                    hsv[..., 1] = 255
                    hsv[..., 0] = ang * 180 / np.pi / 2
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                    writer.write(bgr)


if __name__ == '__main__':
    configure_logging(logging.INFO)
    main(parse_args())
