"""
Inference script
"""
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

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
from nodetracker.node import LightningODEVAE
from nodetracker.utils.logging import save_config

logger = logging.getLogger('TrainScript')


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    logger.info(f'Configuration:\n{OmegaConf.to_yaml(cfg)}')
    raw_cfg = OmegaConf.to_object(cfg)
    cfg = GlobalConfig.from_dict(raw_cfg)

    logs_path = os.path.join(OUTPUTS_PATH, cfg.train.logging_cfg.path)
    logger.info(f'Logs output path: "{logs_path}"')
    logs_config_path = os.path.join(logs_path, cfg.train.logging_cfg.name, f'config-eval-{cfg.eval.experiment}.yaml')
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

    checkpoint_path = os.path.join(logs_path, cfg.eval.checkpoint_path)
    model = LightningODEVAE.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        observable_dim=cfg.model.observable_dim,
        hidden_dim=cfg.model.hidden_dim,
        latent_dim=cfg.model.latent_dim
    )
    accelerator = cfg.resources.accelerator

    # Perform Inference
    mse_func = nn.MSELoss()
    result = []
    metrics = defaultdict(list)

    for bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata in tqdm(test_loader):
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs = [v.to(accelerator) for v in [bboxes_obs, bboxes_unobs, ts_obs, ts_unobs]]
        ts_all = torch.cat([ts_obs, ts_unobs], dim=0)  # FIXME: #1
        bboxes_hat, _, _ = model(bboxes_obs, ts_obs, ts_all)
        bboxes_hat = bboxes_hat[-ts_unobs.shape[0]:, :, :]  # FIXME: #2

        curr_batch_size = bboxes_obs.shape[0]
        for batch_index in range(curr_batch_size):
            mse_val = mse_func(bboxes_unobs, bboxes_hat).detach().item()
            scene_name = metadata['scene_name'][batch_index]
            object_id = metadata['object_id'][batch_index]
            frame_start = metadata['frame_ids'][0][batch_index].detach().item()
            frame_end = metadata['frame_ids'][-1][batch_index].detach().item()
            frame_range = f'{frame_start}-{frame_end}'

            result.append([scene_name, object_id, frame_range, mse_val])
            metrics['MSE'].append(mse_val)

    # Save Inference
    experiment_path = os.path.join(logs_path, cfg.eval.experiment)
    Path(experiment_path).mkdir(parents=True, exist_ok=True)
    inference_fullname = f'{cfg.eval.experiment}_{cfg.eval.model_type}_{cfg.eval.inference_name}'
    inf_result_filename = f'{inference_fullname}-inference.csv'
    inf_result_filepath = os.path.join(experiment_path, inf_result_filename)

    with open(inf_result_filepath, 'w', encoding='utf-8') as f:
        f.write('scene_name,object_id,frame_range,MSE')  # Header
        for r in result:
            scene_name, object_id, frame_range, mse = r
            f.write(f'\n{scene_name},{object_id},{frame_range},{mse:.3f}')

    inf_metrics_filename = f'{inference_fullname}-metrics.json'
    inf_metrics_filepath = os.path.join(experiment_path, inf_metrics_filename)
    metrics = {k: np.array(v).mean() for k, v in metrics.items()}
    with open(inf_metrics_filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    main()
