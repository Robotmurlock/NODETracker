"""
Dataset utils
"""
from collections import defaultdict
from typing import List, Optional, Dict, Union

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

    def __call__(self, items: List[Dict[str, Union[dict, torch.Tensor]]]) \
            -> Dict[str, Union[dict, torch.Tensor]]:
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
        unstack_items: Dict[str, List[Union[dict, torch.Tensor]]] = defaultdict(list)
        for item in items:
            for k, v in item.items():
                unstack_items[k].append(v)
        collated_batch = {}
        for k, v in unstack_items.items():
            if k != 'metadata':
                collated_batch[k] = torch.stack(v, dim=1)
            else:
                collated_batch[k] = {}
                for m_k in v[0].keys():
                    if m_k in ['flow', 'images']:
                        collated_batch[k][m_k] = torch.stack([item[m_k] for item in v], dim=1)
                    else:
                        collated_batch[k][m_k] = default_collate([item[m_k] for item in v])

        x_obs, x_aug_unobs, t_obs, t_unobs = \
            [collated_batch[k] for k in ['t_bboxes_obs', 't_aug_bboxes_unobs', 't_ts_obs', 't_ts_unobs']]

        # Apply augmentations at batch level (optional)
        x_obs, x_aug_unobs, t_obs, t_unobs = self._augmentation(x_obs, x_aug_unobs, t_obs, t_unobs)
        collated_batch['t_bboxes_obs'] = x_obs
        collated_batch['t_aug_bboxes_unobs'] = x_aug_unobs
        collated_batch['t_ts_obs'] = t_obs
        collated_batch['t_ts_unobs'] = t_unobs

        return collated_batch


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
