"""
Any dataset to YOLO format (ultralytics).
"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict

import yaml
from tqdm import tqdm

from nodetracker.common.project import OUTPUTS_PATH
from nodetracker.datasets import dataset_factory
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
    parser.add_argument('--assets-path', type=str, required=True, help='Datasets path.')
    parser.add_argument('--dataset-name', type=str, required=True, help='Dataset name.')
    parser.add_argument('--dataset-type', type=str, required=True, help='Dataset type.')
    parser.add_argument('--split-index-path', type=str, required=True, help='Split index path.')
    parser.add_argument('--output-path', type=str, required=False, default=OUTPUTS_PATH,
                        help='Path where yolo data is stored.')
    parser.add_argument('--include-oov', action='store_true', help='Skip out-of-view objects.')
    parser.add_argument('--include-occ', action='store_true', help='Skip occluded objects.')
    parser.add_argument('--skip-test', action='store_true', help='Skips test split.')
    parser.add_argument('--sampling-step', type=int, required=False, default=1,
                        help='Sample data from sequences. Default: One image per second (with 30 fps assumption)')
    parser.add_argument('--yolo-dataset-name', type=str, required=False, default='YOLO-Dataset', help='Dataset name.')
    parser.add_argument('--category-filter', type=str, required=False, default=None, help='Filter categories.')
    return parser.parse_args()


YOLOAnnotRow = Tuple[int, float, float, float, float]


def parse_annotations(
    frame_index: int,
    objects_data: Dict[str, dict],
    lookup: LookupTable
) -> List[YOLOAnnotRow]:
    """
    Parses annotations for all objects present in the frame. Also updates Lookup table classes.

    Args:
        frame_index: Frame index (not same as frame id) - only required for warning info.
        objects_data: Data for each object present in the frame
        lookup: Lookup table (Note: this function changes state of the lookup table)

    Returns:
        List of YOLO annotations in the frame `frame_index`
    """
    annots: List[YOLOAnnotRow] = []
    for object_id, data in objects_data.items():
        bbox = data['bbox']
        category = data['category']
        x_min, y_min, width, height = bbox

        # Validation
        x_max, y_max = x_min + width, y_min + height
        if not (-0.25 <= x_min <= x_max <= 1.25) or not (-0.25 <= y_min <= y_max <= 1.25):
            logger.warning(f'Invalid bbox coordinates for frame index {frame_index} object {object_id}: '
                           f'{x_min} {y_min} {x_max} {y_max}! Skipping...')
            continue

        # Clip bboxes that are outside of view
        x_min, y_min, x_max, y_max = [min(1.0, max(0.0, v)) for v in [x_min, y_min, x_max, y_max]]

        # Convert to x_center, y_center, width, height
        width, height = x_max - x_min, y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2

        # Get object class index
        class_index = lookup.add(category)

        annots.append((class_index, x_center, y_center, width, height))

    return annots


def main(args: argparse.Namespace) -> None:
    assert args.sampling_step >= 1, f'Invalid value {args.sampling_step} for sampling step!'
    category_filter = args.category_filter.split(',') if args.category_filter is not None else None

    yolo_dataset_path = os.path.join(args.output_path, args.yolo_dataset_name)
    yolo_config_path = os.path.join(yolo_dataset_path, 'config.yaml')
    yolo_lookup_path = os.path.join(yolo_dataset_path, 'lookup.json')

    with open(args.split_index_path, 'r', encoding='utf-8') as f:
        split_index = yaml.safe_load(f)

    lookup = LookupTable(add_unknown_token=False)

    for split in split_index.keys():
        if split == 'test' and args.skip_test:
            logger.warning('Skipping test split!')
            continue

        images_path = os.path.join(yolo_dataset_path, split, 'images')
        annots_path = os.path.join(yolo_dataset_path, split, 'labels')
        image_index_path = os.path.join(yolo_dataset_path, split, 'index.txt')
        Path(images_path).mkdir(parents=True, exist_ok=True)
        Path(annots_path).mkdir(parents=True, exist_ok=True)
        image_index: List[str] = []
        n_total, n_skipped = 0, 0

        dataset_path = os.path.join(args.assets_path, args.dataset_name)
        dataset = dataset_factory(
            name=args.dataset_type,
            path=dataset_path,
            sequence_list=split_index[split],
            future_len=1,  # Not relevant
            history_len=1,  # Not relevant
            additional_params={
                'skip_corrupted': True,
                'allow_missing_annotations': True
            } if args.dataset_type in ['MOT20', 'DanceTrack'] else None
        )
        scene_names = dataset.scenes
        for scene_name in tqdm(scene_names, desc=f'Converting - {split}', unit='scene'):
            scene_info = dataset.get_scene_info(scene_name)
            seqlength = scene_info.seqlength
            object_ids = dataset.get_scene_object_ids(scene_name)

            for i in range(seqlength):
                if (i + 1) % args.sampling_step != 0:
                    continue  # Ignoring frame

                # Fetch info about all objects in the current frame
                objects_data: Dict[str, dict] = {}
                for object_id in object_ids:
                    data = dataset.get_object_data_label_by_frame_index(object_id, i)
                    if data is None:
                        continue  # Object is not present in current frame

                    occ, oov = data['occ'], data['oov']
                    if (not args.include_oov and oov) or (not args.include_occ and occ):
                        continue  # Object is occluded or out of view

                    category = data['category']
                    if category_filter is not None and category not in category_filter:
                        continue  # Not an object of interest

                    objects_data[object_id] = data

                if len(objects_data) == 0:
                    n_skipped += 1
                    continue  # Empty frame

                sample_id = f'{scene_name}_{i:06d}'

                # Save YOLO annotations
                annots = parse_annotations(
                    frame_index=i,
                    objects_data=objects_data,
                    lookup=lookup
                )
                if len(annots) == 0:
                    n_skipped += 1
                    continue  # Empty frame after removing bad bboxes

                annot = '\n'.join([' '.join(str(v) for v in a) for a in annots])
                annot_path = os.path.join(annots_path, f'{sample_id}.txt')
                with open(annot_path, 'w', encoding='utf-8') as f:
                    f.write(annot)

                # Save image (create symlink)
                src_image_path = dataset.get_scene_image_path(scene_name, frame_index=i)
                _, image_extension = os.path.splitext(src_image_path)
                dst_image_path = os.path.join(images_path, f'{sample_id}{image_extension}')
                os.symlink(src_image_path, dst_image_path)
                assert os.path.exists(dst_image_path), f'Failed to create symlink "{dst_image_path}".'
                image_index.append(dst_image_path)
                n_total += 1

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
