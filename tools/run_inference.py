"""
Inference script
"""
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Dict, Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nodetracker.common.project import CONFIGS_PATH, ASSETS_PATH, OUTPUTS_PATH
from nodetracker.config_parser import GlobalConfig
from nodetracker.datasets.mot import TorchMOTTrajectoryDataset
from nodetracker.datasets.utils import ode_dataloader_collate_func
from nodetracker.node import load_or_create_model
from nodetracker.utils.logging import save_config

logger = logging.getLogger('TrainScript')


@torch.no_grad()
def run_inference(model: nn.Module, accelerator: str, data_loader: DataLoader) -> Tuple[List[list], Dict[str, Any]]:
    """
    Performs inference for given model and dataset

    Args:
        model: Model which is used to perform inference
        accelerator: CPU/GPU
        data_loader: Dataset data loader

    Returns:
        Result for each sample in dataset, Aggregated dataset metrics
    """
    mse_func = nn.MSELoss()
    result = []
    metrics = defaultdict(list)

    model.eval()
    for bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata in tqdm(data_loader, unit='sample', desc='Running inference'):
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs = [v.to(accelerator) for v in [bboxes_obs, bboxes_unobs, ts_obs, ts_unobs]]
        bboxes_unobs_hat, *_ = model(bboxes_obs, ts_obs, ts_unobs)

        curr_batch_size = bboxes_obs.shape[0]
        for batch_index in range(curr_batch_size):
            mse_val = mse_func(bboxes_unobs, bboxes_unobs_hat).detach().item()
            scene_name = metadata['scene_name'][batch_index]
            object_id = metadata['object_id'][batch_index]
            frame_start = metadata['frame_ids'][0][batch_index].detach().item()
            frame_end = metadata['frame_ids'][-1][batch_index].detach().item()
            frame_range = f'{frame_start}-{frame_end}'

            result.append([scene_name, object_id, frame_range, mse_val])
            metrics['MSE'].append(mse_val)

    return result, metrics


def save_inference(
    result: List[list],
    metrics: Dict[str, Any],
    logs_path: str,
    model_type: str,
    dataset_name: str,
    experiment_name: str,
    inference_name: str
) -> None:
    """
    Save inference results (dataset inference and evaluation metrics)

    Args:
        result: Inference results
        metrics: Evaluation metrics
        logs_path: Logs path
        model_type: Model type (architecture)
        dataset_name: Dataset name
        experiment_name: Experiment (concrete model) name
        inference_name: Inference name (name of the inference run)
    """
    inference_fullname = f'{model_type}-{dataset_name}-{experiment_name}_{inference_name}'
    inference_dirpath = os.path.join(logs_path, 'inferences', inference_fullname)
    Path(inference_dirpath).mkdir(parents=True, exist_ok=True)

    inf_result_filepath = os.path.join(inference_dirpath, 'inference.csv')

    with open(inf_result_filepath, 'w', encoding='utf-8') as f:
        f.write('scene_name,object_id,frame_range,MSE')  # Header
        for r in result:
            scene_name, object_id, frame_range, mse = r
            f.write(f'\n{scene_name},{object_id},{frame_range},{mse:.3f}')

    inf_metrics_filepath = os.path.join(inference_dirpath, 'metrics.json')
    metrics = {k: np.array(v).mean() for k, v in metrics.items()}
    with open(inf_metrics_filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f)


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    logger.info(f'Configuration:\n{OmegaConf.to_yaml(cfg)}')
    raw_cfg = OmegaConf.to_object(cfg)
    cfg = GlobalConfig.from_dict(raw_cfg)

    logs_path = os.path.join(OUTPUTS_PATH, cfg.dataset.name, cfg.train.experiment)
    logger.info(f'Logs output path: "{logs_path}"')
    logs_config_path = os.path.join(logs_path, f'config-inference.yaml')
    save_config(raw_cfg, logs_config_path)

    dataset_test_path = os.path.join(ASSETS_PATH, cfg.dataset.test_path)
    logger.info(f'Dataset train path: "{dataset_test_path}".')

    test_dataset = TorchMOTTrajectoryDataset(
        path=dataset_test_path,
        history_len=cfg.dataset.history_len,
        future_len=cfg.dataset.future_len
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        collate_fn=ode_dataloader_collate_func,
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.resources.num_workers,
        shuffle=True
    )

    checkpoint_path = os.path.join(logs_path, 'checkpoints', cfg.eval.checkpoint)
    model = load_or_create_model(model_type=cfg.model.type, params=cfg.model.params, checkpoint_path=checkpoint_path)
    accelerator = cfg.resources.accelerator

    inf_result, eval_metrics = run_inference(
        model=model,
        accelerator=accelerator,
        data_loader=test_loader
    )
    save_inference(
        result=inf_result,
        metrics=eval_metrics,
        logs_path=logs_path,
        dataset_name=cfg.dataset.name,
        experiment_name=cfg.eval.experiment,
        model_type=cfg.model.type,
        inference_name=cfg.eval.inference_name
    )


if __name__ == '__main__':
    main()
