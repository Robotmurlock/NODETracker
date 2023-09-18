"""
Converts MOT20 format to "NODE-MOT" format using symlinks for folder. Also creates `.split_index.yaml` file.
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List

import yaml

from nodetracker.utils import file_system


def parse_args() -> argparse.Namespace:
    """
    Conversion configuration. Run example:

    python3 nodetracker/dataset/tools/script_mot_to_nodemot_format.py \
        --input {INPUT_DATASET_PATH} \
        --output {OUTPUT_DATASET_PATH}

    Returns:
        Parsed configuration.
    """
    parser = argparse.ArgumentParser(description='Conversion configuration.')
    parser.add_argument('--input', type=str, required=True, help='Original dataset path.')
    parser.add_argument('--output', type=str, required=True, help='New dataset path with symlinks.')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    input_path: str = args.input
    output_path: str = args.output
    assert not os.path.exists(output_path), f'Path "{output_path}" is already taken!'

    # Check that input contains all splits
    input_dirnames = set(file_system.listdir(input_path))
    assert {'train', 'val', 'test'}.issubset(input_dirnames)

    # Create "new" dataset
    Path(output_path).mkdir(parents=True, exist_ok=True)

    split_index: Dict[str, List[str]] = {}
    for dirname in input_dirnames:
        input_split_path = os.path.join(input_path, dirname)
        scene_names = file_system.listdir(input_split_path)
        split_index[dirname] = scene_names

        for scene_name in scene_names:
            src_path = os.path.join(input_split_path, scene_name)
            dst_path = os.path.join(output_path, scene_name)
            os.symlink(src_path, dst_path)

    # Save split index
    split_index_path = os.path.join(output_path, '.split_index.yaml')
    with open(split_index_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(split_index, f)


if __name__ == '__main__':
    main(parse_args())
