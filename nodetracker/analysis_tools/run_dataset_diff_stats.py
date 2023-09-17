"""
Diff stats analysis
"""
import json
import logging
from pathlib import Path
from collections import defaultdict
import yaml

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from nodetracker.common import conventions
from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets import TorchTrajectoryDataset, dataset_factory
from nodetracker.datasets import transforms
from nodetracker.utils import pipeline

logger = logging.getLogger('DatasetDiffStats')


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, _ = pipeline.preprocess(cfg, name='dataset_diff_stats')

    dataset = TorchTrajectoryDataset(
        dataset_factory(
            name=cfg.dataset.name,
            path=cfg.dataset.fullpath,
            sequence_list=cfg.dataset.split_index['train'],
            history_len=cfg.dataset.history_len,
            future_len=cfg.dataset.future_len,
            additional_params=cfg.dataset.additional_params
        ),
        transform=transforms.BBoxRelativeToLastObsTransform(),
        # augmentation_before_transform=cfg.augmentations.before_transform,
        # augmentation_after_transform=cfg.augmentations.after_transform
    )

    category_data = defaultdict(lambda: {
        'sum_bbox': torch.zeros(4, dtype=torch.float32),
        'sum_bbox2': torch.zeros(4, dtype=torch.float32),
        'n_total': 0
    })

    sum_bbox = torch.zeros(4, dtype=torch.float32)  # Used to calculate mean and std
    sum_bbox2 = torch.zeros(4, dtype=torch.float32)  # Used to calculate std
    n_total = 0

    # noinspection PyTypeChecker
    for sample in tqdm(dataset, unit='sample', desc='Calculating diff statistics'):
        bboxes_obs, bboxes_unobs, _, _, _, _, _, metadata = sample.values()
        category = metadata['category']

        for bboxes in [bboxes_obs, bboxes_unobs]:
            # General stats
            sum_bbox += bboxes.sum(dim=0)
            sum_bbox2 += torch.pow(bboxes, 2).sum(dim=0)
            n_total += bboxes.shape[0]

            # Category stats
            category_data[category]['sum_bbox'] += bboxes.sum(dim=0)
            category_data[category]['sum_bbox2'] += torch.pow(bboxes, 2).sum(dim=0)
            category_data[category]['n_total'] += bboxes.shape[0]

    # Calculate statistics (global)
    mean_bbox = sum_bbox / n_total
    mean_bbox2 = sum_bbox2 / n_total
    std_bbox = (mean_bbox2 - mean_bbox ** 2) ** 0.5

    # Convert all to lists
    mean_bbox, mean_bbox2, std_bbox = [v.numpy().tolist() for v in [mean_bbox, mean_bbox2, std_bbox]]

    # Calculate statistics (category)
    for cat_data in category_data.values():
        cat_data['mean'] = cat_data['sum_bbox'] / cat_data['n_total']
        cat_data['mean2'] = cat_data['sum_bbox2'] / cat_data['n_total']
        cat_data['std'] = (cat_data['mean2'] - cat_data['mean'] ** 2) ** 0.5

        del cat_data['n_total']
        del cat_data['sum_bbox']
        del cat_data['sum_bbox2']
        del cat_data['mean2']

        cat_data['mean'] = cat_data['mean'].numpy().tolist()
        cat_data['std'] = cat_data['std'].numpy().tolist()


    stats_data = {
        'mean': mean_bbox,
        'std': std_bbox,
        'by_category': dict(category_data)
    }

    logger.info(f'Stats\n{yaml.dump(stats_data)}')

    output_path = conventions.get_analysis_filepath(cfg.path.master, 'first_order_difference_stats.json')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f)


if __name__ == '__main__':
    main()
