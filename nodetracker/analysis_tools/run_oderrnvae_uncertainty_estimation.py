"""
Script for uncertainty estimation for ODERNNVAE.
"""
import logging
import math
import os
from typing import List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from nodetracker.common import conventions
from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets import transforms
from nodetracker.node import LightningODERNNVAE
from nodetracker.node import load_or_create_model
from nodetracker.node.utils.autoregressive import AutoregressiveForecasterDecorator
from nodetracker.utils import pipeline
from tools.utils import create_mot20_dataloader

logger = logging.getLogger('VisualizeTrajectories')


def plot_trajectories(
    indices: np.ndarray,
    traj_means: np.ndarray,
    traj_stds: np.ndarray,
    coord_names: List[str],
    ylim: Optional[Tuple] = None
) -> plt.Figure:
    """
    Plots trajectories to visualize their approximated distributions.

    Args:
        indices: Indices (time points)
        traj_means: Trajectory means
        traj_stds: Trajectory stds (required for confidence interval)
        coord_names: Coordinate names
        ylim: Plot coord range

    Returns:
        Figure
    """
    traj_conf_interval_lower_bound = traj_means - traj_stds
    traj_conf_interval_upper_bound = traj_means + traj_stds
    print(traj_means.shape, traj_stds.shape)

    n_coords = traj_means.shape[1]
    n_rows = 4
    n_cols = math.ceil(n_coords / n_rows)
    fig, axs = plt.subplots(figsize=(5*n_cols, 12), nrows=n_rows, ncols=n_cols)
    for i, coord_name in enumerate(coord_names):
        ri = i % n_rows
        ci = i // n_rows
        ax = axs[ri][ci] if n_cols > 1 else axs[ri]

        ax.plot(indices, traj_means[:, i], color='red')
        ax.fill_between(indices, traj_conf_interval_lower_bound[:, i],
                        traj_conf_interval_upper_bound[:, i], color='red', alpha=0.1)
        ax.scatter(indices, traj_means[:, i], color='red', s=16)
        ax.set_xlabel('t')
        ax.set_ylabel(coord_name)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid()
        plt.tight_layout()
    return fig


@torch.no_grad()
def run_odernnvae_sampling_inference_and_visualization(
    model: LightningODERNNVAE,
    accelerator: str,
    data_loader: DataLoader,
    transform: transforms.InvertibleTransform,
    n_samples: int,
    output_path: str
) -> None:
    """
    Performs inference with Monte Carlo mean and std estimation

    Args:
        model: Model which is used to perform inference
        accelerator: CPU/GPU
        data_loader: Dataset data loader
        transform: Invert (training) transformations
        n_samples:
        output_path: Path where figure is stored
    """
    model.eval()

    for bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, _ in tqdm(data_loader, unit='sample', desc='Running inference'):
        # `t` prefix means that tensor is mapped to transformed space
        t_bboxes_obs, _, t_ts_obs, t_ts_unobs = transform.apply([bboxes_obs, bboxes_unobs, ts_obs, ts_unobs],
                                                                shallow=False)  # preprocess
        t_bboxes_obs, t_ts_obs, t_ts_unobs = [v.to(accelerator) for v in [t_bboxes_obs, t_ts_obs, t_ts_unobs]]
        t_bboxes_unobs_hat_grouped = model.predict_monte_carlo(t_bboxes_obs, t_ts_obs, t_ts_unobs, n_samples=n_samples)

        # Postprocess is now more complicated
        # Old values are expanded for inverse transformation
        # Postprocess has to be performed on CPU
        t_bboxes_unobs_hat_grouped = t_bboxes_unobs_hat_grouped.detach().cpu()
        bboxes_obs = bboxes_obs.detach().cpu()

        unobs_time_len, batch_size, group_size, _ = t_bboxes_unobs_hat_grouped.shape
        t_bboxes_unobs_hat_grouped_flat = t_bboxes_unobs_hat_grouped.view(unobs_time_len, batch_size, -1)
        obs_time_len, _, _ = bboxes_obs.shape
        bboxes_obs_replicated = bboxes_obs.unsqueeze(-1).repeat(1, 1, 1, n_samples).view(obs_time_len, batch_size, -1)
        # noinspection PyArgumentList
        _, bboxes_unobs_hat_grouped_flat, *_ = \
            transform.inverse([bboxes_obs_replicated, t_bboxes_unobs_hat_grouped_flat], n_samples=n_samples)
        bboxes_unobs_hat_grouped = bboxes_unobs_hat_grouped_flat.view(unobs_time_len, batch_size, group_size, -1)

        # Estimate mean and std
        # Mean and std have to be estimated at this point since it is hard run inverse transform on std
        bboxes_unobs_hat_mean = bboxes_unobs_hat_grouped.mean(dim=2)
        bboxes_unobs_hat_std = bboxes_unobs_hat_grouped.std(dim=2)
        print(bboxes_unobs_hat_grouped)
        print(bboxes_unobs_hat_mean)
        print(bboxes_unobs)

        bboxes_unobs_hat_mean = bboxes_unobs_hat_mean.numpy()
        bboxes_unobs_hat_std = bboxes_unobs_hat_std.numpy()
        bboxes_unobs_hat_mean = bboxes_unobs_hat_mean[:, 0, :]
        bboxes_unobs_hat_std = bboxes_unobs_hat_std[:, 0, :]

        indices = ts_unobs[:, 0, 0].numpy()
        fig = plot_trajectories(
            indices=indices,
            traj_means=bboxes_unobs_hat_mean,
            traj_stds=bboxes_unobs_hat_std,
            coord_names=['x', 'y', 'w', 'h'],
            ylim=(-5, 5)
        )
        fig.savefig(f'{output_path}_bboxes_mean_std.png')
        break


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='visualize_trajectories')

    postprocess_transform = transforms.transform_factory(cfg.transform.name, cfg.transform.params)

    # Load dataset for visualization
    dataset_path = os.path.join(cfg.path.assets, cfg.dataset.get_split_path(cfg.eval.split))
    logger.info(f'Dataset {cfg.eval.split} path: "{dataset_path}".')
    data_loader = create_mot20_dataloader(
        dataset_path=dataset_path,
        cfg=cfg,
        postprocess_transform=postprocess_transform,
        shuffle=False,
        batch_size=1
    )

    checkpoint_path = conventions.get_checkpoint_path(experiment_path, cfg.eval.checkpoint) \
        if cfg.eval.checkpoint else None
    model = load_or_create_model(
        model_type=cfg.model.type,
        params=cfg.model.params,
        checkpoint_path=checkpoint_path
    )
    assert isinstance(model, LightningODERNNVAE), \
        'Uncertainty measurement is only available for ODERNNVAE!'

    model = AutoregressiveForecasterDecorator(model, keep_history=cfg.eval.autoregressive_keep_history) \
        if cfg.eval.autoregressive else model
    accelerator = cfg.resources.accelerator
    model.to(accelerator)

    run_odernnvae_sampling_inference_and_visualization(
        model=model,
        accelerator=accelerator,
        data_loader=data_loader,
        transform=postprocess_transform,
        n_samples=4,
        output_path=os.path.join(cfg.path.master, f'{cfg.eval.experiment}_odernnvae_monte_carlo_estimate'),
    )


if __name__ == '__main__':
    main()
