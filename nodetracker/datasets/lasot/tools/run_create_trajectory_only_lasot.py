"""
This script creates smaller sized LaSOT dataset that can be used
to train models that use only trajectories as inputs (no images).
"""
import argparse
import logging
import os
import shutil
from pathlib import Path

from tqdm import tqdm

from nodetracker.utils import file_system
from nodetracker.utils.logging import configure_logging

logger = logging.getLogger('CreateLasotTrajectoryDataset')


def parse_args() -> argparse.Namespace:
    """
    Configuration for trajectory only LaSOT creation.

    Returns:
        Parsed configuration
    """
    parser = argparse.ArgumentParser(description='Trajectory only LaSOT creation configuration.')
    parser.add_argument('--input', type=str, required=True, help='Path to the original dataset')
    parser.add_argument('--output', type=str, required=True, help='Path to the new dataset')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    EXPECTED_SCENE_FILES = {
        'full_occlusion.txt',
        'groundtruth.txt',
        'img',
        'out_of_view.txt'
    }

    input_path: str = args.input
    output_path: str = args.output

    logger.info(f'Input path is "{input_path}".')
    scenes = file_system.listdir(input_path)
    n_scenes = len(scenes)
    logger.info(f'Number of scenes is {n_scenes}.')
    for scene_name in tqdm(scenes, unit='scene', total=n_scenes, desc='Copying scenes'):
        scene_path = os.path.join(input_path, scene_name)
        scene_files = set(file_system.listdir(scene_path))
        assert EXPECTED_SCENE_FILES.issubset(scene_files), f'[{scene_name}]: {scene_files} != {EXPECTED_SCENE_FILES}'

        img_dirpath = os.path.join(scene_path, 'img')
        image_filename = sorted(file_system.listdir(img_dirpath))[0]
        image_relpath = os.path.join('img', image_filename)
        scene_output_path = os.path.join(output_path, scene_name)
        files_to_copy = [
            'full_occlusion.txt',
            'groundtruth.txt',
            image_relpath,
            'out_of_view.txt'
        ]
        for filename in files_to_copy:
            src_path = os.path.join(scene_path, filename)
            dst_path = os.path.join(scene_output_path, filename)
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_path, dst_path)



if __name__ == '__main__':
    configure_logging()
    main(parse_args())