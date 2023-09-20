"""
Visualizes dataset scene annotations.
"""
import argparse
import logging
import os
import random
from pathlib import Path

import cv2
from tqdm import tqdm

from nodetracker.datasets.factory import dataset_factory
from nodetracker.library.cv.bbox import BBox
from nodetracker.library.cv.video_writer import MP4Writer
from nodetracker.utils.logging import configure_logging

logger = logging.getLogger('VisualizeDatasetScene')


def parse_args() -> argparse.Namespace:
    """
    Dataset scene visualization configuration. Run example:

    python3

    Returns:
        Parsed configuration
    """
    parser = argparse.ArgumentParser(description='Scene visualization configuration.')
    parser.add_argument('--assets-path', type=str, required=True, help='Datasets path.')
    parser.add_argument('--dataset-name', type=str, required=True, help='Dataset name.')
    parser.add_argument('--dataset-type', type=str, required=True, help='Dataset type.')
    parser.add_argument('--output-path', type=str, required=False, default='tmp_scene_viz', help='Output path.')
    parser.add_argument('--n-scenes', type=int, required=False, default=1, help='Number of datasets to sample.')
    parser.add_argument('--fps', type=int, required=False, default=30, help='Video output fps.')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    output_path = args.output_path
    Path(output_path).mkdir(parents=True, exist_ok=True)

    dataset_path = os.path.join(args.assets_path, args.dataset_name)
    dataset = dataset_factory(
        name=args.dataset_type,
        path=dataset_path,
        future_len=1,  # Not relevant
        history_len=1,  # Not relevant
        additional_params={
            'skip_corrupted': True,
            'allow_missing_annotations': True
        } if args.dataset_type in ['MOT20', 'DanceTrack'] else None
    )

    scene_names = dataset.scenes
    random.shuffle(scene_names)
    scene_names = scene_names[:args.n_scenes]
    for scene_name in tqdm(scene_names, desc='Creating scene videos', unit='scene'):
        object_ids = dataset.get_scene_object_ids(scene_name)
        scene_length = dataset.get_scene_info(scene_name).seqlength

        scene_video_path = os.path.join(output_path, f'{scene_name}.mp4')
        with MP4Writer(scene_video_path, fps=args.fps) as mp4_writer:
            for i in tqdm(range(scene_length), desc=f'Visualizing "{scene_name}"', unit='frame'):
                image_path = dataset.get_scene_image_path(scene_name, i)
                frame = cv2.imread(image_path)
                assert frame is not None, f'Failed to load image for frame {i} on scene "{scene_name}" with path "{image_path}"!'

                for object_id in object_ids:
                    data = dataset.get_object_data_label_by_frame_index(object_id, frame_index=i)
                    if data is None:
                        # Missing object data (possibly occluded or oov)
                        continue

                    bbox = BBox.from_yxwh(*data['bbox'], clip=True)
                    frame = bbox.draw(frame)

                mp4_writer.write(frame)

        logger.info(f'Saved scene visualization at path "{scene_video_path}"')


if __name__ == '__main__':
    configure_logging()
    main(parse_args())
