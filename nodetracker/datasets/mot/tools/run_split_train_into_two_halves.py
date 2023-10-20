"""
Performs original "train" dataset split into halves where
the first half belongs to the train
and the second half belongs to the validation.

Steps:
1. Parse seqinfo file and extract seqlength
2. Split gt/gt.txt and det/det.txt (csv files)
3. Copy images (or create symlinks)
4. Create new seqinfo files
"""
import argparse
import configparser
import logging
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from nodetracker.utils import file_system
from nodetracker.utils.logging import configure_logging

logger = logging.getLogger('MOT20Split')


def parse_args() -> argparse.Namespace:
    """
    Scene split parse configuration.

    Example run:
        python3 nodetracker/datasets/mot/tools/run_split_train_into_two_halves.py \
            --input-path {ASSETS_PATH}/MOT17/train \
            --output-path {ASSETS_PATH}/MOT17

    Returns:
        Scene split configuration
    """
    parser = argparse.ArgumentParser(description='Split MOT configuration')
    parser.add_argument('--input-path', type=str, required=True, help='MOT dataset path.')
    parser.add_argument('--output-path', type=str, required=True, help='MOT new dataset path')
    return parser.parse_args()


def parse_ini(path: str, subject: str) -> dict:
    """
    Parse an .ini file and return a dictionary of the specified section.

    Args:
        path: The file path of the .ini file to be read.
        subject: The section in the .ini file to be converted to a dictionary.

    Returns:
        A dictionary containing the key-value pairs of the specified section
        in the .ini file.
    """
    raw_info = configparser.ConfigParser()
    raw_info.read(path)
    return dict(raw_info[subject])


def save_ini(data: dict, path: str, subject: str) -> None:
    """
    Save a dictionary into an .ini file.

    Args:
        data: The dictionary to be saved.
        path: The file path of the .ini file.
        subject: The header under which the dictionary will be saved.
    """
    config = configparser.ConfigParser()
    config[subject] = data

    with open(path, 'w', encoding='utf-8') as configfile:
        config.write(configfile)


def split_scene(input_path: str, output_path: str, scene: str) -> None:
    """
    Split a MOT scene into two halves and create new folders containing the respective information.

    Args:
        input_path: The directory containing the MOT scene.
        output_path: The directory where the two new scenes will be created.
        scene: The name of the scene.
    """
    scene_path = os.path.join(input_path, scene)
    seqinfo = parse_ini(os.path.join(scene_path, 'seqinfo.ini'), 'Sequence')
    seqlength = int(seqinfo['seqlength'])
    middle_frame = seqlength // 2

    # Create new directories for the two halves
    first_scene_name = scene + '-H1'
    second_scene_name = scene + '-H2'
    first_half_path = os.path.join(output_path, 'train', first_scene_name)
    second_half_path = os.path.join(output_path, 'val', second_scene_name)

    Path(first_half_path).mkdir(parents=True, exist_ok=True)
    Path(second_half_path).mkdir(parents=True, exist_ok=True)

    # Split gt.txt and det.txt using pandas
    for file in ['gt/gt.txt', 'det/det.txt']:
        df = pd.read_csv(os.path.join(scene_path, file), header=None)
        first_half = df[df[0] <= middle_frame]
        second_half = df[df[0] > middle_frame].copy()
        second_half[0] = second_half[0] - middle_frame

        first_half_filepath = os.path.join(first_half_path, file)
        Path(first_half_filepath).parent.mkdir(exist_ok=True, parents=True)
        first_half.to_csv(first_half_filepath, header=None, index=False)

        second_half_filepath = os.path.join(second_half_path, file)
        Path(second_half_filepath).parent.mkdir(exist_ok=True, parents=True)
        second_half.to_csv(second_half_filepath, header=None, index=False)

    # Write new seqinfo.ini files for the two halves
    seqinfo.pop('seqlength')
    seqinfo.pop('name')
    first_half_seqinfo = {**seqinfo, 'seqlength': str(middle_frame), 'name': first_scene_name}
    second_half_seqinfo = {**seqinfo, 'seqlength': str(seqlength - middle_frame), 'name': second_scene_name}

    save_ini(first_half_seqinfo, os.path.join(first_half_path, 'seqinfo.ini'), 'Sequence')
    save_ini(second_half_seqinfo, os.path.join(second_half_path, 'seqinfo.ini'), 'Sequence')

    # Create symlinks for images
    img_path = os.path.join(scene_path, 'img1')
    for i in range(1, seqlength + 1):
        source_img = os.path.join(img_path, f'{i:06}.jpg')
        dst_img = os.path.join(first_half_path, 'img1', f'{i:06}.jpg') if i <= middle_frame \
            else os.path.join(second_half_path, 'img1', f'{i - middle_frame:06}.jpg')
        Path(dst_img).parent.mkdir(exist_ok=True, parents=True)
        os.symlink(source_img, dst_img)


def main(args: argparse.Namespace) -> None:
    input_path: str = args.input_path
    output_path: str = args.output_path

    scenes = file_system.listdir(input_path)
    n_scenes = len(scenes)
    logger.info(f'Indexed {n_scenes} scenes: {scenes}.')

    for scene in tqdm(scenes, total=len(scenes), unit='scene'):
        split_scene(
            scene=scene,
            input_path=input_path,
            output_path=output_path
        )


if __name__ == '__main__':
    configure_logging()
    main(parse_args())
