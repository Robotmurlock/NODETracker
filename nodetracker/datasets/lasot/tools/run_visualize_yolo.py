"""
YOLO model visualization on LaSOT dataset.
"""
import argparse
import logging
import os
import re
from pathlib import Path

import cv2
import ultralytics
from tqdm import tqdm

from nodetracker.common.project import OUTPUTS_PATH
from nodetracker.datasets.lasot.core import LaSOTDataset
from nodetracker.library.cv.video_writer import MP4Writer
from nodetracker.utils.logging import configure_logging

logger = logging.getLogger('YOLOv8Visualization')


def parse_args() -> argparse.Namespace:
    """
    LaSOT to YOLOv8.

    Returns:
        Parsed configuration.
    """
    parser = argparse.ArgumentParser(description='Sequence analysis - visualization')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--model-path', type=str, required=True, help='Path where trained model is stored.')
    parser.add_argument('--output-path', type=str, required=False, default=OUTPUTS_PATH,
                        help='Path where yolo data is stored.')
    parser.add_argument('--scene-filter', type=str, required=True, help='Sequence filter - regex (not optional)')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    scene_filter: str = args.scene_filter
    dataset = LaSOTDataset(args.dataset_path, history_len=1, future_len=1)
    yolo = ultralytics.YOLO(args.model_path)

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    scene_names = [scene_name for scene_name in dataset.scenes if re.match(scene_filter, scene_name)]
    for scene_name in scene_names:
        video_path = os.path.join(args.output_path, f'{scene_name}.mp4')
        with MP4Writer(video_path, fps=30) as writer:
            for object_id in dataset.get_scene_object_ids(scene_name):
                length = dataset.get_object_data_length(object_id)
                for i in tqdm(range(length)):
                    data = dataset.get_object_data_label_by_frame_index(object_id, i)
                    img_path = data['image_path']
                    # noinspection PyUnresolvedReferences
                    img = cv2.imread(img_path)
                    assert img is not None, f'Failed to load image "{img_path}"!'

                    prediction_raw = yolo.predict(img)[0]
                    img = prediction_raw.plot()

                    writer.write(img)


if __name__ == '__main__':
    configure_logging(logging.INFO)
    main(parse_args())
