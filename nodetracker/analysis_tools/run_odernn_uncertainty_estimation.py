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
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nodetracker.common import conventions
from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets import transforms
from nodetracker.node import LightningODERNN
from nodetracker.node import load_or_create_model
from nodetracker.utils import pipeline
from tools.utils import create_mot20_dataloader

logger = logging.getLogger('VisualizeTrajectories')


def plot_trajectory_estimation(
    indices: np.ndarray,
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
        traj_means: Trajectory means
        traj_stds: Trajectory stds (required for confidence interval)
        gt_traj: Ground truth trajectory
        coord_names: Coordinate names
        loss: GaussianNLLLoss

    Returns:
        Figure
    """
    # Calculate (-std, +std) confidence interval
    traj_conf_interval_lower_bound = traj_means - traj_stds
    traj_conf_interval_upper_bound = traj_means + traj_stds

    n_cols = len(coord_names)

    # Centering graph around ground truth
    center_traj = gt_traj.mean(dim=0)
    ylims = [(c - 0.01, c + 0.01) for c in center_traj]

    fig, axs = plt.subplots(figsize=(4, 6), nrows=1, ncols=n_cols)
    for col in range(n_cols):
        ax = axs[col]
        ax.grid()
        coord_name = coord_names[col]

        # Plot estimation (TODO: Duplicate code)
        ax.plot(indices, traj_means[:, col], color='red', zorder=-1)
        ax.fill_between(indices, traj_conf_interval_lower_bound[:, col],
                             traj_conf_interval_upper_bound[:, col], color='red', alpha=0.1, zorder=-1)
        ax.scatter(indices, traj_means[:, col], color='red', s=16, zorder=-1)

        # Plot GT trajectory
        ax.plot(indices, gt_traj[:, col], color='k', zorder=1, label='gt')
        ax.scatter(indices, gt_traj[:, col], color='k', s=32, marker='x', zorder=1, label='gt')

        # Set graph items
        ax.set_title(f'[{coord_name}] Trajectory estimation (nll={loss:.2f})')
        ax.set_xlabel('t')
        ax.set_ylabel(coord_name)
        ax.set_ylim(ylims[col])
        ax.legend()
        plt.tight_layout()

    return fig


@torch.no_grad()
def run_odernn_model_gaussian_inference_and_visualization(
    model: LightningODERNN,
    accelerator: str,
    data_loader: DataLoader,
    transform: transforms.InvertibleTransformWithStd,
    experiment_name: str,
    max_plots: int,
    visualize: bool
) -> None:
    """
    Performs inference with ODERNN that models gaussian distribution.

    Args:
        model: Model which is used to perform inference
        accelerator: CPU/GPU
        data_loader: Dataset data loader
        transform: Invert (training) transformations
        experiment_name: Name of the ODERNN experiment
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
        t_bboxes_unobs_hat_mean, t_bboxes_unobs_hat_std, *_ = model.inference(t_bboxes_obs, t_ts_obs, t_ts_unobs)

        # Move tensors to cpu
        bboxes_obs = bboxes_obs.detach().cpu()
        t_bboxes_unobs_hat_mean = t_bboxes_unobs_hat_mean.detach().cpu()
        t_bboxes_unobs_hat_std = t_bboxes_unobs_hat_std.detach().cpu()

        # Postprocess MEAN (trivial)
        _, bboxes_unobs_hat_mean, *_ = transform.inverse([bboxes_obs, t_bboxes_unobs_hat_mean])
        # Postprocess STD (not trivial) - depends on the transform function
        bboxes_unobs_hat_std = transform.inverse_std(t_bboxes_unobs_hat_std)

        # Calculate loss (TODO: Duplicate code)
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
            gt_traj=bboxes_unobs,
            coord_names=coord_names,
            loss=loss.item()
        )
        Path(experiment_name).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(experiment_name, f'{traj_index + 1:04d}.png'))

    logger.info(f'Average Gaussian negative log likelihood is: {loss_sum.item() / n_batches:.2f}')


# Parameters (improvisation)
VISUALIZE = False
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
        postprocess_transform=None,  # Preprocessing and postprocessing are applied manually
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
    assert isinstance(model, LightningODERNN), 'Uncertainty measurement is only available for ODERNN!'
    assert model.is_modeling_gaussian, 'Uncertainty measurement is only available for ODERNN trained with GaussianNLLLoss!'
    accelerator = cfg.resources.accelerator
    model.to(accelerator)

    experiments_dirname = 'odernn_guassian_modeling'
    run_odernn_model_gaussian_inference_and_visualization(
        model=model,
        accelerator=accelerator,
        data_loader=data_loader,
        transform=postprocess_transform,
        experiment_name=os.path.join(cfg.path.master, experiments_dirname, cfg.eval.experiment),
        max_plots=MAX_PLOTS,
        visualize=VISUALIZE
    )


if __name__ == '__main__':
    main()
