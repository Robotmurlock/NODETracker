"""
LaSOT to YOLO format (ultralytics)
"""
import argparse
import logging
import os
from pathlib import Path
import json
from typing import List

import yaml
from tqdm import tqdm

from nodetracker.common.project import OUTPUTS_PATH
from nodetracker.datasets.lasot.core import LaSOTDataset
from nodetracker.utils.logging import configure_logging
from nodetracker.utils.lookup import LookupTable

logger = logging.getLogger('LaSOTtoYOLOv8')


def parse_args() -> argparse.Namespace:
    """
    LaSOT to YOLOv8.

    Returns:
        Parsed configuration.
    """
    parser = argparse.ArgumentParser(description='Sequence analysis - visualization')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--split-index-path', type=str, required=True, help='Split index path.')
    parser.add_argument('--output-path', type=str, required=False, default=OUTPUTS_PATH,
                        help='Path where yolo data is stored.')
    parser.add_argument('--skip-oov', action='store_false', help='Skip out-of-view objects.')
    parser.add_argument('--skip-occ', action='store_false', help='Skip occluded objects.')
    parser.add_argument('--sampling-step', type=int, required=False, default=30,
                        help='Sample data from sequences. Default: One image per second (with 30 fps assumption)')
    parser.add_argument('--dataset-name', type=str, required=False, default='YOLO-Dataset', help='Dataset name.')
    parser.add_argument('--category-filter', type=str, required=False, default=None, help='Filter categories.')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    assert args.sampling_step >= 1, f'Invalid value {args.sampling_step} for sampling step!'
    category_filter = args.category_filter.split(',') if args.category_filter is not None else None

    yolo_dataset_path = os.path.join(args.output_path, args.dataset_name)
    yolo_config_path = os.path.join(yolo_dataset_path, 'config.yaml')
    yolo_lookup_path = os.path.join(yolo_dataset_path, 'lookup.json')

    with open(args.split_index_path, 'r', encoding='utf-8') as f:
        split_index = yaml.safe_load(f)

    lookup = LookupTable(add_unknown_token=False)

    for split in split_index.keys():
        images_path = os.path.join(yolo_dataset_path, split, 'images')
        annots_path = os.path.join(yolo_dataset_path, split, 'labels')
        image_index_path = os.path.join(yolo_dataset_path, split, 'index.txt')
        Path(images_path).mkdir(parents=True, exist_ok=True)
        Path(annots_path).mkdir(parents=True, exist_ok=True)
        image_index: List[str] = []
        n_total, n_skipped = 0, 0

        dataset = LaSOTDataset(args.dataset_path, history_len=1, future_len=1, sequence_list=split_index[split])
        scene_names = dataset.scenes
        for scene_name in tqdm(scene_names, desc=f'Converting - {split}', unit='scene'):
            object_ids = dataset.get_scene_object_ids(scene_name)

            for object_id in object_ids:
                n_data_points = dataset.get_object_data_length(object_id)
                for index in range(n_data_points):
                    if (index + 1) % args.sampling_step != 0:
                        continue

                    data = dataset.get_object_data_label(object_id, index)

                    # Extract data
                    bbox = data['bbox']
                    category = data['category']
                    if category_filter is not None and category not in category_filter:
                        continue

                    frame_id = data['frame_id']
                    image_path = data['image_path']
                    occ, oov = data['occ'], data['oov']
                    index = lookup.add(category)
                    x_min, y_min, width, height = bbox
                    _, image_extension = os.path.splitext(image_path)
                    sample_id = f'{object_id}_{frame_id}'
                    if (args.skip_oov and oov) or (args.skip_occ and occ):
                        n_skipped += 1
                        continue

                    # Validation
                    x_max, y_max = x_min + width, y_min + height
                    if not (0 <= x_min <= x_max <= 1) or not (0 <= y_min <= y_max <= 1):
                        logger.warning(f'Invalid bbox coordinates {sample_id}: '
                                       f'{x_min} {y_min} {x_max} {y_max}! Skipping...')
                        n_skipped += 1
                        continue

                    n_total += 1

                    # Create image symlink
                    new_image_name = f'{sample_id}{image_extension}'
                    new_image_path = os.path.join(images_path, new_image_name)
                    os.symlink(image_path, new_image_path)
                    assert os.path.exists(new_image_path), f'Failed to create symlink "{new_image_path}".'
                    image_index.append(new_image_path)

                    # Write annotation
                    x_center = x_min + width / 2
                    y_center = y_min + height / 2
                    annot_list = [index, x_center, y_center, width, height]
                    annot = ' '.join([str(x) for x in annot_list])
                    annot_name = f'{sample_id}.txt'
                    annot_path = os.path.join(annots_path, annot_name)

                    with open(annot_path, 'w', encoding='utf-8') as f:
                        f.write(annot)

        logger.info(f'Total number of samples for split "{split}" is {n_total}. '
                    f'Number of skipped samples is {n_skipped}')

        with open(image_index_path, 'w', encoding='utf-8') as f:
            f.write(os.linesep.join(image_index))

    logger.info(f'Created YOLO dataset at path "{yolo_dataset_path}"')

    # Create config
    config = {
        'path': yolo_dataset_path,
        'nc': len(lookup),
        'names': lookup.tokens
    }
    for split in split_index:
        config[split] = f'{split}/index.txt'

    logger.info(f'Config:\n{yaml.safe_dump(config)}')
    with open(yolo_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f)
    logger.info(f'Saved YOLO config to "{yolo_config_path}"')

    with open(yolo_lookup_path, 'w', encoding='utf-8') as f:
        json.dump(lookup.serialize(), f, indent=2)
    logger.info(f'Saved YOLO lookup table to "{yolo_lookup_path}"')


if __name__ == '__main__':
    configure_logging(logging.INFO)
    main(parse_args())

