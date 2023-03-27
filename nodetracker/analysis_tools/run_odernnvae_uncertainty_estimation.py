"""
Script for uncertainty estimation for ODERNNVAE.
"""
import logging
import os
from pathlib import Path
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import get_cmap
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nodetracker.common import conventions
from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets import transforms
from nodetracker.node import LightningODERNNVAE
from nodetracker.node import load_or_create_model
from nodetracker.utils import pipeline
from tools.utils import create_mot20_dataloader

logger = logging.getLogger('VisualizeTrajectories')


def plot_trajectory_estimation(
    indices: np.ndarray,
    trajs: np.ndarray,
    traj_means: np.ndarray,
    traj_stds: np.ndarray,
    gt_traj: np.ndarray,
    coord_names: List[str],
    loss: float
) -> plt.Figure:
    """
    Plots sampled trajectories (up) and estimated trajectory (down).

    Args:
        indices: Indices (time points)
        trajs: Sampled trajectories
        traj_means: Trajectory means
        traj_stds: Trajectory stds (required for confidence interval)
        gt_traj: Ground truth trajectory
        coord_names: Coordinate names
        loss: GaussianNLLLoss

    Returns:
        Figure
    """
    n_samples = trajs.shape[1]
    n_cols = trajs.shape[2]  # ~ n_coords
    colors = get_cmap('Accent').colors
    n_colors = len(colors)

    # Calculate (-std, +std) confidence interval
    traj_conf_interval_lower_bound = traj_means - traj_stds
    traj_conf_interval_upper_bound = traj_means + traj_stds

    # Centering graph around ground truth
    center_traj = gt_traj.mean(dim=0)
    ylims = [(c - 0.01, c + 0.01) for c in center_traj]

    fig, axs = plt.subplots(figsize=(4 * n_cols, 6), nrows=2, ncols=n_cols)
    for col in range(n_cols):
        ax_up, ax_down = axs[0][col], axs[1][col]  # up - sampled trajectories, down - estimated trajectories
        ax_up.grid(), ax_down.grid()
        coord_name = coord_names[col]

        # Plot sampled trajectories
        for sample_index in range(n_samples):
            color = colors[sample_index % n_colors]

            traj = trajs[:, sample_index, col]
            ax_up.plot(indices, traj, color=color, zorder=-1)
            ax_up.scatter(indices, traj, color=color, s=16, zorder=-1)

        # Plot estimation
        ax_down.plot(indices, traj_means[:, col], color='red', zorder=-1)
        ax_down.fill_between(indices, traj_conf_interval_lower_bound[:, col],
                             traj_conf_interval_upper_bound[:, col], color='red', alpha=0.1, zorder=-1)
        ax_down.scatter(indices, traj_means[:, col], color='red', s=16, zorder=-1)

        # Plot parts that are same for both plots
        for ax in [ax_up, ax_down]:
            # Plot GT trajectory
            ax.plot(indices, gt_traj[:, col], color='k', zorder=1, label='gt')
            ax.scatter(indices, gt_traj[:, col], color='k', s=32, marker='x', zorder=1, label='gt')

            # Set graph items
            ax.set_xlabel('t')
            ax.set_ylabel(coord_name)
            ax.set_ylim(ylims[col])
            ax.legend()

        # Set titles
        ax_up.set_title(f'[{coord_name}] Sampled trajectories')
        ax_down.set_title(f'[{coord_name}] Trajectory estimation (nll={loss:.2f})')
        plt.tight_layout()

    return fig


@torch.no_grad()
def run_odernnvae_sampling_inference_and_visualization(
    model: LightningODERNNVAE,
    accelerator: str,
    data_loader: DataLoader,
    transform: transforms.InvertibleTransform,
    n_samples: int,
    sampling_exp_name: str,
    max_plots: int,
    visualize: bool
) -> None:
    """
    Performs inference with Monte Carlo mean and std estimation

    Args:
        model: Model which is used to perform inference
        accelerator: CPU/GPU
        data_loader: Dataset data loader
        transform: Invert (training) transformations
        n_samples:
        sampling_exp_name: Name of the ODERNNVAE sampling experiment
        max_plots: Number of plots
        visualize: If visualization then perform few inferences plot and stop
            else run full evaluation
    """
    model.eval()
    loss_func = nn.GaussianNLLLoss()
    loss_sum = torch.tensor(0, dtype=torch.float32)
    n_batches = 0

    # Constants
    coord_names = ['x', 'y', 'w', 'h']

    traj_index = 0  # Plot index
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
        bboxes_obs_replicated = bboxes_obs.unsqueeze(-2).repeat(1, 1, n_samples, 1).view(obs_time_len, batch_size, -1)
        # noinspection PyArgumentList
        _, bboxes_unobs_hat_grouped_flat, *_ = \
            transform.inverse([bboxes_obs_replicated, t_bboxes_unobs_hat_grouped_flat], n_samples=n_samples)
        bboxes_unobs_hat_grouped = bboxes_unobs_hat_grouped_flat.view(unobs_time_len, batch_size, group_size, -1)

        # Estimate mean and std
        # Mean and std have to be estimated at this point since it is hard run inverse transform on std
        bboxes_unobs_hat_mean = bboxes_unobs_hat_grouped.mean(dim=-2)
        bboxes_unobs_hat_std = bboxes_unobs_hat_grouped.std(dim=-2)

        # Calculate loss
        loss = loss_func(bboxes_unobs_hat_mean, bboxes_unobs, bboxes_unobs_hat_std)
        n_batches += 1
        loss_sum += loss

        traj_index += 1
        if traj_index >= max_plots:
            if visualize:
                break
            else:
                continue

        # Remove batch size
        bboxes_unobs_hat_mean = bboxes_unobs_hat_mean[:, 0, :]
        bboxes_unobs_hat_std = bboxes_unobs_hat_std[:, 0, :]
        bboxes_unobs = bboxes_unobs[:, 0, :]

        # Convert to numpy for visualization
        bboxes_unobs_hat_mean = bboxes_unobs_hat_mean.numpy()
        bboxes_unobs_hat_std = bboxes_unobs_hat_std.numpy()

        # Visualize
        indices = ts_unobs[:, 0, 0].numpy()
        fig = plot_trajectory_estimation(
            indices=indices,
            traj_means=bboxes_unobs_hat_mean,
            traj_stds=bboxes_unobs_hat_std,
            trajs=bboxes_unobs_hat_grouped[:, 0, :, :],
            gt_traj=bboxes_unobs,
            coord_names=coord_names,
            loss=loss.item()
        )
        output_dirpath = f'{sampling_exp_name}_{n_samples}'
        Path(output_dirpath).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(output_dirpath, f'{traj_index + 1:04d}.png'))

    logger.info(f'Average Gaussian negative log likelihood is: {loss_sum.item() / n_batches:.2f}')


# Parameters (improvisation)
VISUALIZE = False
N_SAMPLES = 5
MAX_PLOTS = 10


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
        train=False,
        batch_size=1 if VISUALIZE else None
    )

    checkpoint_path = conventions.get_checkpoint_path(experiment_path, cfg.eval.checkpoint) \
        if cfg.eval.checkpoint else None
    model = load_or_create_model(
        model_type=cfg.model.type,
        params=cfg.model.params,
        checkpoint_path=checkpoint_path
    )
    assert isinstance(model, LightningODERNNVAE), 'Uncertainty measurement is only available for ODERNNVAE!'
    accelerator = cfg.resources.accelerator
    model.to(accelerator)

    experiments_dirname = 'odernnvae_sampling_experiments'
    full_exp_name = f'{cfg.eval.experiment}_odernnvae_monte_carlo_estimate'
    run_odernnvae_sampling_inference_and_visualization(
        model=model,
        accelerator=accelerator,
        data_loader=data_loader,
        transform=postprocess_transform,
        n_samples=N_SAMPLES,
        sampling_exp_name=os.path.join(cfg.path.master, experiments_dirname, full_exp_name),
        max_plots=MAX_PLOTS,
        visualize=VISUALIZE
    )


if __name__ == '__main__':
    main()
