"""
Training script
"""
import json
import logging
from pathlib import Path

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
        ),
        transform=transforms.BBoxRelativeToLastObsTransform(),
        augmentation_before_transform=cfg.augmentations.before_transform,
        augmentation_after_transform=cfg.augmentations.after_transform
    )

    sum_bbox = torch.zeros(4, dtype=torch.float32)  # Used to calculate mean and std
    sum_bbox2 = torch.zeros(4, dtype=torch.float32)  # Used to calculate std
    n_total = 0
    # noinspection PyTypeChecker
    for bboxes_obs, bboxes_unobs, _, _, _ in tqdm(dataset, unit='sample', desc='Calculating diff statistics'):
        for data in [bboxes_obs, bboxes_unobs]:
            sum_bbox += data.sum(dim=0)
            sum_bbox2 += torch.pow(data, 2).sum(dim=0)
            n_total += data.shape[0]

    # Calculate statistics
    mean_bbox = sum_bbox / n_total
    mean_bbox2 = sum_bbox2 / n_total
    std_bbox = (mean_bbox2 - mean_bbox ** 2) ** 0.5

    # Convert all to lists
    mean_bbox, mean_bbox2, std_bbox = [v.numpy().tolist() for v in [mean_bbox, mean_bbox2, std_bbox]]

    logger.info(f'Mean={mean_bbox}, Std={std_bbox}')

    stats_data = {
        'mean': mean_bbox,
        'std': std_bbox
    }
    output_path = conventions.get_analysis_filepath(cfg.path.master, 'first_order_difference_stats.json')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f)


if __name__ == '__main__':
    main()
