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
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nodetracker.common import conventions
from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets.mot import TorchMOTTrajectoryDataset
from nodetracker.datasets.utils import ode_dataloader_collate_func
from nodetracker.node import load_or_create_model
from nodetracker.utils import pipeline

logger = logging.getLogger('InferenceScript')


_CONFIG_SAVE_FILENAME = 'config-eval.yaml'


@torch.no_grad()
def run_inference(model: nn.Module, accelerator: str, data_loader: DataLoader) -> Tuple[List[list], List[list], Dict[str, Any]]:
    """
    Performs inference for given model and dataset

    Args:
        model: Model which is used to perform inference
        accelerator: CPU/GPU
        data_loader: Dataset data loader

    Returns:
        - Predictions for each sample in dataset
        - Metrics for each sample on dataset
        - Aggregated (averaged) dataset metrics
    """
    mse_func = nn.MSELoss()
    predictions = []
    sample_metrics = []
    dataset_metrics = defaultdict(list)

    model.eval()
    for bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata in tqdm(data_loader, unit='sample', desc='Running inference'):
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs = [v.to(accelerator) for v in [bboxes_obs, bboxes_unobs, ts_obs, ts_unobs]]
        bboxes_unobs_hat, *_ = model(bboxes_obs, ts_obs, ts_unobs)

        curr_obs_time_len = bboxes_obs.shape[0]
        curr_unobs_time_len, curr_batch_size = bboxes_unobs.shape[:2]
        for batch_index in range(curr_batch_size):
            scene_name = metadata['scene_name'][batch_index]
            object_id = metadata['object_id'][batch_index]
            frame_start = metadata['frame_ids'][0][batch_index].detach().item()
            middle_frame_id = frame_start + curr_obs_time_len  # first unobs frame
            frame_end = metadata['frame_ids'][-1][batch_index].detach().item()
            frame_range = f'{frame_start}-{middle_frame_id}-{frame_end}'
            sample_id = [scene_name, object_id, frame_range]

            # Save sample predictions (inference) with gt values
            # format: ymin, xmin, w, h
            for frame_relative_index in range(curr_unobs_time_len):
                frame_id = middle_frame_id + frame_relative_index
                prediction_id = sample_id + [frame_id]
                bboxes_unobs_gt_list = bboxes_unobs[0, batch_index, :].detach().cpu().numpy().tolist()
                bboxes_unobs_pred_list = bboxes_unobs_hat[0, batch_index, :].detach().cpu().numpy().tolist()
                pred = prediction_id + bboxes_unobs_pred_list + bboxes_unobs_gt_list
                predictions.append(pred)

            # Save sample eval metrics
            # format: mse
            mse_val = mse_func(bboxes_unobs, bboxes_unobs_hat).detach().item()
            s_metrics = sample_id + [mse_val]
            sample_metrics.append(s_metrics)

            # Save sample eval metrics for aggregation
            dataset_metrics['MSE'].append(mse_val)

    dataset_metrics = {k: np.array(v).mean() for k, v in dataset_metrics.items()}
    return predictions, sample_metrics, dataset_metrics


def save_inference(
    predictions: List[list],
    sample_metrics: List[list],
    dataset_metrics: Dict[str, Any],
    experiment_path: str,
    model_type: str,
    dataset_name: str,
    split: str,
    experiment_name: str,
    inference_name: str
) -> None:
    """
    Save inference results (dataset inference and evaluation metrics)

    Args:
        predictions: Inference predictions
        sample_metrics: Sample level metrics
        dataset_metrics: Dataset level metrics (aggregated)
        experiment_path: Model experiment path
        model_type: Model type (architecture)
        dataset_name: Dataset name
        split: Split name
        experiment_name: Experiment (concrete model) name
        inference_name: Inference name (name of the inference run)
    """
    inference_fullname = conventions.get_inference_fullname(model_type, dataset_name, split, experiment_name, inference_name)
    inference_dirpath = os.path.join(experiment_path, conventions.INFERENCES_DIRNAME, inference_fullname)
    Path(inference_dirpath).mkdir(parents=True, exist_ok=True)

    # Save predictions
    inf_predictions_filepath = os.path.join(inference_dirpath, 'inference.csv')
    with open(inf_predictions_filepath, 'w', encoding='utf-8') as f:
        f.write('scene_name,object_id,frame_range,frame_id,p_ymin,p_xmin,p_w,p_h,gt_ymin,gt_xmin,gt_w,gt_h')  # Header
        for p in predictions:
            scene_name, object_id, frame_range, frame_id, *coords = p
            coords_str = ','.join([f'{c:.4f}' for c in coords])
            f.write(f'\n{scene_name},{object_id},{frame_range},{frame_id},{coords_str}')

    # Save sample evaluation metrics
    inf_sample_metrics_filepath = os.path.join(inference_dirpath, 'sample_metrics.csv')
    with open(inf_sample_metrics_filepath, 'w', encoding='utf-8') as f:
        f.write('scene_name,object_id,frame_range,MSE')  # Header
        for sm in sample_metrics:
            scene_name, object_id, frame_range, mse = sm
            f.write(f'\n{scene_name},{object_id},{frame_range},{mse:.4f}')

    # Save dataset evaluation (aggregated) metrics
    inf_dataset_metrics_filepath = os.path.join(inference_dirpath, 'dataset_metrics.json')
    with open(inf_dataset_metrics_filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset_metrics, f, indent=2)


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='inference')

    dataset_path = os.path.join(cfg.path.assets, cfg.dataset.get_split_path(cfg.eval.split))
    logger.info(f'Dataset {cfg.eval.split} path: "{dataset_path}".')

    dataset = TorchMOTTrajectoryDataset(
        path=dataset_path,
        history_len=cfg.dataset.history_len,
        future_len=cfg.dataset.future_len
    )

    data_loader = DataLoader(
        dataset=dataset,
        collate_fn=ode_dataloader_collate_func,
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.resources.num_workers,
        shuffle=True
    )

    checkpoint_path = conventions.get_checkpoint_path(experiment_path, cfg.eval.checkpoint)
    model = load_or_create_model(model_type=cfg.model.type, params=cfg.model.params, checkpoint_path=checkpoint_path)
    accelerator = cfg.resources.accelerator

    inf_predictions, eval_sample_metrics, eval_dataset_metrics = run_inference(
        model=model,
        accelerator=accelerator,
        data_loader=data_loader
    )
    save_inference(
        predictions=inf_predictions,
        sample_metrics=eval_sample_metrics,
        dataset_metrics=eval_dataset_metrics,
        experiment_path=experiment_path,
        dataset_name=cfg.dataset.name,
        split=cfg.eval.split,
        experiment_name=cfg.eval.experiment,
        model_type=cfg.model.type,
        inference_name=cfg.eval.inference_name
    )


if __name__ == '__main__':
    main()
