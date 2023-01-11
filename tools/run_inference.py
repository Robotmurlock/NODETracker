"""
Inference script
"""
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Dict, Any, Iterator

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nodetracker.common import conventions
from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets import transforms
from nodetracker.datasets.mot import TorchMOTTrajectoryDataset
from nodetracker.datasets.utils import ode_dataloader_collate_func
from nodetracker.node import load_or_create_model
from nodetracker.utils import pipeline

logger = logging.getLogger('InferenceScript')


_CONFIG_SAVE_FILENAME = 'config-eval.yaml'


@torch.no_grad()
def run_inference(
    model: nn.Module,
    accelerator: str,
    data_loader: DataLoader,
    postprocess_transform: transforms.InvertibleTransform,
    chunk_size: int = 1000
) -> Iterator[Tuple[bool, List[list], List[list], Dict[str, Any]]]:
    """
    Performs inference for given model and dataset

    Args:
        model: Model which is used to perform inference
        accelerator: CPU/GPU
        data_loader: Dataset data loader
        postprocess_transform: Invert (training) transformations
        chunk_size: Important in case predictions for all samples can't fit in ram (chunking)

    Returns:
        - Predictions for each sample in dataset
        - Metrics for each sample on dataset
        - Aggregated (averaged) dataset metrics
    """
    mse_func = nn.MSELoss()
    model.eval()

    n_batch_steps = max(1, chunk_size // data_loader.batch_size)
    predictions = []
    sample_metrics = []
    dataset_metrics = defaultdict(list)
    batch_cnt = 0
    first_chunk = True

    for bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata in tqdm(data_loader, unit='sample', desc='Running inference'):
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs = [v.to(accelerator) for v in [bboxes_obs, bboxes_unobs, ts_obs, ts_unobs]]
        bboxes_unobs_hat, *_ = model(bboxes_obs, ts_obs, ts_unobs)
        _, bboxes_unobs_hat, *_ = postprocess_transform.inverse([bboxes_obs, bboxes_unobs_hat])

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
                bboxes_unobs_gt_list = bboxes_unobs[frame_relative_index, batch_index, :].detach().cpu().numpy().tolist()
                bboxes_unobs_pred_list = bboxes_unobs_hat[frame_relative_index, batch_index, :].detach().cpu().numpy().tolist()
                pred = prediction_id + bboxes_unobs_pred_list + bboxes_unobs_gt_list
                predictions.append(pred)

            # Save sample eval metrics
            # format: mse
            mse_val = mse_func(bboxes_unobs, bboxes_unobs_hat).detach().item()
            s_metrics = sample_id + [mse_val]
            sample_metrics.append(s_metrics)

            # Save sample eval metrics for aggregation
            dataset_metrics['MSE'].append(mse_val)

        batch_cnt += 1
        if batch_cnt >= n_batch_steps:
            # Chunk is full -> yield it
            dataset_metrics = {k: np.array(v).mean() for k, v in dataset_metrics.items()}
            yield first_chunk, predictions, sample_metrics, dataset_metrics

            first_chunk = False

            # Empty memory
            predictions = []
            sample_metrics = []
            dataset_metrics = defaultdict(list)
            batch_cnt = 0

    if batch_cnt >= 1:
        # Yield only in case there are unsaved items
        dataset_metrics = {k: np.array(v).mean() for k, v in dataset_metrics.items()}
        yield first_chunk, predictions, sample_metrics, dataset_metrics


def save_dataset_metrics(
    dataset_chunked_metrics: List[Dict[str, Any]],
    inference_dirpath: str
) -> None:
    """
    Save global dataset metrics.
    Note: Metrics should be "average friendly".

    Args:
        dataset_chunked_metrics: Dataset metrics for each inference chunk
        inference_dirpath: Path where to store inference data
    """
    # Transform: List[Dict[str, float]] -> Dict[str, List[float]]
    metric_names = list(dataset_chunked_metrics[0].keys())
    dataset_metrics = {mn: [] for mn in metric_names}
    for chunk in dataset_chunked_metrics:
        for mn in metric_names:
            dataset_metrics[mn].append(chunk[mn])

    # Aggregate
    dataset_metrics = {k: sum(v) / len(v) for k, v in dataset_metrics.items()}

    # Save dataset evaluation (aggregated) metrics
    inf_dataset_metrics_filepath = os.path.join(inference_dirpath, 'dataset_metrics.json')
    with open(inf_dataset_metrics_filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset_metrics, f, indent=2)


def save_inference(
    predictions: List[list],
    sample_metrics: List[list],
    inference_dirpath: str,
    append: bool = False
) -> None:
    """
    Save inference results (dataset inference and evaluation metrics)

    Args:
        predictions: Inference predictions
        sample_metrics: Sample level metrics
        inference_dirpath: Path where inference data is stored
        append: Append to existing file
    """
    Path(inference_dirpath).mkdir(parents=True, exist_ok=True)
    write_mode = 'a' if append else 'w'

    # Save predictions
    inf_predictions_filepath = os.path.join(inference_dirpath, 'inference.csv')
    with open(inf_predictions_filepath, write_mode, encoding='utf-8') as f:
        if not append:
            # Header
            f.write('scene_name,object_id,frame_range,frame_id,p_ymin,p_xmin,p_w,p_h,gt_ymin,gt_xmin,gt_w,gt_h')

        # data
        for p in predictions:
            scene_name, object_id, frame_range, frame_id, *coords = p
            coords_str = ','.join([f'{c:.4f}' for c in coords])
            f.write(f'\n{scene_name},{object_id},{frame_range},{frame_id},{coords_str}')

    # Save sample evaluation metrics
    inf_sample_metrics_filepath = os.path.join(inference_dirpath, 'sample_metrics.csv')
    with open(inf_sample_metrics_filepath, write_mode, encoding='utf-8') as f:
        if not append:
            # Header
            f.write('scene_name,object_id,frame_range,MSE')

        for sm in sample_metrics:
            scene_name, object_id, frame_range, mse = sm
            f.write(f'\n{scene_name},{object_id},{frame_range},{mse:.4f}')


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='inference')

    dataset_path = os.path.join(cfg.path.assets, cfg.dataset.get_split_path(cfg.eval.split))
    logger.info(f'Dataset {cfg.eval.split} path: "{dataset_path}".')

    postprocess_transform = transforms.transform_factory(cfg.transform.name, cfg.transform.params)
    dataset = TorchMOTTrajectoryDataset(
        path=dataset_path,
        history_len=cfg.dataset.history_len,
        future_len=cfg.dataset.future_len,
        postprocess=postprocess_transform
    )

    data_loader = DataLoader(
        dataset=dataset,
        collate_fn=ode_dataloader_collate_func,
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.resources.num_workers,
        shuffle=True
    )

    checkpoint_path = conventions.get_checkpoint_path(experiment_path, cfg.eval.checkpoint) if cfg.eval.checkpoint else None
    model = load_or_create_model(model_type=cfg.model.type, params=cfg.model.params, checkpoint_path=checkpoint_path)
    accelerator = cfg.resources.accelerator
    model.to(accelerator)

    inference_dirpath = conventions.get_inference_path(
        experiment_path=experiment_path,
        model_type=cfg.model.type,
        dataset_name=cfg.dataset.name,
        split=cfg.eval.split,
        experiment_name=cfg.eval.experiment,
        inference_name=cfg.eval.inference_name
    )

    dataset_chunked_metrics = []
    for first_chunk, inf_predictions, eval_sample_metrics, eval_dataset_metrics \
            in run_inference(model, accelerator, data_loader, postprocess_transform=postprocess_transform):
        save_inference(
            predictions=inf_predictions,
            sample_metrics=eval_sample_metrics,
            inference_dirpath=inference_dirpath,
            append=not first_chunk
        )

        dataset_chunked_metrics.append(eval_dataset_metrics)

    save_dataset_metrics(
        dataset_chunked_metrics,
        inference_dirpath
    )


if __name__ == '__main__':
    main()
