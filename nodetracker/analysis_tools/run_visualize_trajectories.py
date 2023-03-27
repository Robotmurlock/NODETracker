"""
Script for latent and prediction trajectory analysis.
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
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nodetracker.common import conventions
from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets import transforms
from nodetracker.node import LightningODERNN, LightningODERNNVAE
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
def run_visualize_trajectory_analysis(
    model: nn.Module,
    accelerator: str,
    data_loader: DataLoader,
    transform: transforms.InvertibleTransform,
    output_path: str,
    batch_size: int,
    model_latent_dim: int,
    period: float = 0.25,
    start: float = -1.0,
    end: float = 5.0,
) -> None:
    """
    Visualizes trajectories.

    Args:
        model: Model which is used to perform inference
        accelerator: CPU/GPU
        data_loader: Dataset data loader
        transform: Invert (training) transformations
        output_path: Path where figure is stored
        batch_size: Loader batch size
        model_latent_dim: Model latent trajectory dimension
        period: period between two trajectory time points
        start: trajectory start time point
        end: trajectory end time point
    """
    model.eval()

    ts_unobs = torch.arange(start, end, period).view(-1, 1, 1).repeat(1, batch_size, 1)  # Override ts_unobs
    bboxes_unobs_hat_mean_sum = np.zeros(shape=(ts_unobs.shape[0], 4))
    bboxes_unobs_hat_std_sum = np.zeros(shape=(ts_unobs.shape[0], 4))
    latent_representation_mean_sum = np.zeros(shape=(ts_unobs.shape[0], model_latent_dim))
    latent_representation_std_sum = np.zeros(shape=(ts_unobs.shape[0], model_latent_dim))
    n_batches = 0

    for bboxes_obs, bboxes_unobs, ts_obs, _, _ in tqdm(data_loader, unit='sample', desc='Running inference'):
        ts_unobs = ts_unobs[:, :bboxes_unobs.shape[1], :]  # Cut ts_unobs for last batch

        # `t` prefix means that tensor is mapped to transformed space
        t_bboxes_obs, _, t_ts_obs, t_ts_unobs = transform.apply([bboxes_obs, bboxes_unobs, ts_obs, ts_unobs],
                                                                shallow=False)  # preprocess
        t_bboxes_obs, t_ts_obs, t_ts_unobs = [v.to(accelerator) for v in [t_bboxes_obs, t_ts_obs, t_ts_unobs]]
        t_bboxes_unobs_hat, latent_representation, *_ = model(t_bboxes_obs, t_ts_obs, t_ts_unobs)  # inference
        # In case of multiple suffix values output (tuple) ignore everything except first output
        t_bboxes_unobs_hat = t_bboxes_unobs_hat.detach().cpu()
        latent_representation = latent_representation.detach().cpu()
        _, bboxes_unobs_hat, *_ = transform.inverse([bboxes_obs, t_bboxes_unobs_hat])  # postprocess

        bboxes_unobs_hat = bboxes_unobs_hat.numpy()
        latent_representation = latent_representation.numpy()
        bboxes_unobs_hat_mean_sum += bboxes_unobs_hat.mean(axis=1)
        bboxes_unobs_hat_std_sum += bboxes_unobs_hat.std(axis=1)
        latent_representation_mean_sum += latent_representation.mean(axis=1)
        latent_representation_std_sum += latent_representation.std(axis=1)
        n_batches += 1

    indices = ts_unobs[:, 0, 0].numpy()
    bboxes_unobs_hat_mean = bboxes_unobs_hat_mean_sum / n_batches
    bboxes_unobs_hat_std = bboxes_unobs_hat_std_sum / n_batches

    fig = plot_trajectories(
        indices=indices,
        traj_means=bboxes_unobs_hat_mean,
        traj_stds=bboxes_unobs_hat_std,
        coord_names=['x', 'y', 'w', 'h'],
        ylim=(0, 1)
    )
    fig.savefig(f'{output_path}_bboxes.png')

    latent_representation_mean = latent_representation_mean_sum / n_batches
    latent_representation_std = latent_representation_std_sum / n_batches

    fig = plot_trajectories(
        indices=indices,
        traj_means=latent_representation_mean,
        traj_stds=latent_representation_std,
        coord_names=[f'L-{i}' for i in range(model_latent_dim)],
        ylim=(-5, 5)
    )
    fig.savefig(f'{output_path}_latent_representation.png')


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
        transform=None,  # Preprocessing and postprocessing are applied manually
        shuffle=True,
        train=False
    )

    checkpoint_path = conventions.get_checkpoint_path(experiment_path, cfg.eval.checkpoint) \
        if cfg.eval.checkpoint else None
    model = load_or_create_model(
        model_type=cfg.model.type,
        params=cfg.model.params,
        checkpoint_path=checkpoint_path
    )
    assert isinstance(model, (LightningODERNN, LightningODERNNVAE)), \
        'Visualization currently only supported for ODERNN and ODERNNVAE!'

    model = AutoregressiveForecasterDecorator(model, keep_history=cfg.eval.autoregressive_keep_history) \
        if cfg.eval.autoregressive else model
    accelerator = cfg.resources.accelerator
    model.to(accelerator)

    run_visualize_trajectory_analysis(
        model=model,
        accelerator=accelerator,
        data_loader=data_loader,
        transform=postprocess_transform,
        output_path=os.path.join(cfg.path.master, f'{cfg.eval.experiment}_mean_trajectory'),
        batch_size=cfg.eval.batch_size,
        model_latent_dim=cfg.model.params['hidden_dim']
    )


if __name__ == '__main__':
    main()
