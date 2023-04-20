"""
Dataset utils
"""
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import default_collate

from nodetracker.datasets.augmentations.trajectory import TrajectoryAugmentation, IdentityAugmentation


class OdeDataloaderCollateFunctional:
    """
    ODE Dataloader collate func wrapper that supports configurable augmentations.
    """
    def __init__(self, augmentation: Optional[TrajectoryAugmentation] = None):
        """
        Creates `ode_dataloader_collate_func` wrapper by adding the augmentations after collate function

        Args:
            augmentation: Augmentation applied after collate function is applied (optional)

        Returns:
            `ode_dataloader_collate_func` with (optional) augmentations.
        """
        self._augmentation = augmentation
        if self._augmentation is None:
            self._augmentation = IdentityAugmentation()

    def __call__(self, items: List[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        ODE's collate func: Standard way to batch sequences of dimension (T, *shape)
        where T is time dimension and shape is feature dimension is to create batch
        of size (B, T, *shape) but for NODE it makes more sense to do it as (T, B, *shape)
        which requires custom collate_func

        Args:
            items: Items gathered from WeatherDataset

        Returns:
            collated tensors
        """
        x_obss, t_obss, x_unobss, t_unobss, metadata = zip(*items)
        x_obs, t_obs, x_unobs, t_unobs = [torch.stack(v, dim=1) for v in [x_obss, t_obss, x_unobss, t_unobss]]
        metadata = default_collate(metadata)

        # Apply augmentations at batch level (optional)
        x_obs, t_obs, x_unobs, t_unobs = self._augmentation(x_obs, t_obs, x_unobs, t_unobs)

        return x_obs, t_obs, x_unobs, t_unobs, metadata


def preprocess_batch(batch: tuple) -> tuple:
    """
    Unpacks batch and creates full trajectory tensors

    Args:
        batch: Raw batch

    Returns:
        Preprocessing batch
    """
    bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, metadata = batch
    ts_all = torch.cat([ts_obs, ts_unobs], dim=0)
    bboxes_all = torch.cat([bboxes_obs, bboxes_unobs], dim=0)
    return bboxes_obs, bboxes_unobs, bboxes_all, ts_obs, ts_unobs, ts_all, metadata


def split_trajectory_observed_unobserved(frame_ids: List[int], bboxes: np.ndarray, history_len: int):
    """
    Splits trajectory time points and bboxes int observed (input) trajectory
    and unobserved (ground truth) trajectory.

    Args:
        frame_ids: Full trajectory frame ids
        bboxes: Full trajectory bboxes
        history_len: Observed trajectory length

    Returns:
        - Observed trajectory bboxes
        - Unobserved trajectory bboxes
        - Observed trajectory time points
        - Unobserved trajectory time points
    """

    # Time points
    frame_ts = np.array(frame_ids, dtype=np.float32)
    frame_ts = frame_ts - frame_ts[0] + 1  # Transforming to relative time values
    frame_ts = np.expand_dims(frame_ts, -1)

    # Observed - Unobserved
    bboxes_obs = bboxes[:history_len]
    bboxes_unobs = bboxes[history_len:]
    frame_ts_obs = frame_ts[:history_len]
    frame_ts_unobs = frame_ts[history_len:]

    return bboxes_obs, bboxes_unobs, frame_ts_obs, frame_ts_unobs
