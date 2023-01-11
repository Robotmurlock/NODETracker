"""
Training script
"""
import json
import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from nodetracker.common.project import CONFIGS_PATH
from nodetracker.common import conventions
from nodetracker.datasets import TorchMOTTrajectoryDataset
from nodetracker.datasets import transforms
from nodetracker.utils import pipeline

logger = logging.getLogger('DatasetDiffStats')


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='visualize')

    dataset_train_path = os.path.join(cfg.path.assets, cfg.dataset.train_path)
    logger.info(f'Dataset train path: "{dataset_train_path}".')

    dataset = TorchMOTTrajectoryDataset(
        path=dataset_train_path,
        history_len=cfg.dataset.history_len,
        future_len=cfg.dataset.future_len,
        postprocess=transforms.BboxFirstOrderDifferenceTransform()
    )

    sum_x = 0
    sum_x2 = 0
    n_total = 0
    for bboxes_obs, bboxes_unobs, _, _, _ in tqdm(dataset, unit='sample', desc='Calculating diff statistics'):
        for data in [bboxes_obs, bboxes_unobs]:
            sum_x += data.sum().item()
            sum_x2 += torch.pow(data, 2).sum().item()
            n_total += data.view(-1).shape[0]

    mean_x = sum_x / n_total
    mean_x2 = sum_x2 / n_total
    std_x = (mean_x2 - mean_x ** 2) ** 0.5

    logger.info(f'Mean={mean_x}, Std={std_x}')

    stats_data = {
        'mean': mean_x,
        'std': std_x
    }
    output_path = conventions.get_analysis_filepath(cfg.path.master, 'first_order_difference_stats.json')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f)


if __name__ == '__main__':
    main()
