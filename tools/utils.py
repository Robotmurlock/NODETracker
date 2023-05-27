"""
Utility functions (often used functions)
"""
from typing import Optional, List
from torch import nn

from torch.utils.data import DataLoader

from nodetracker.config_parser import GlobalConfig
from nodetracker.datasets import TorchTrajectoryDataset, dataset_factory
from nodetracker.datasets.augmentations import TrajectoryAugmentation
from nodetracker.datasets.transforms import InvertibleTransform
from nodetracker.datasets.utils import OdeDataloaderCollateFunctional
from nodetracker.common import conventions
from nodetracker.node.factory import load_or_create_model
from nodetracker.node.utils.autoregressive import AutoregressiveForecasterDecorator


def create_dataloader(
    cfg: GlobalConfig,
    split: str,
    train: bool,
    transform: Optional[InvertibleTransform] = None,
    augmentation_before_transform: Optional[TrajectoryAugmentation] = None,
    augmentation_after_transform: Optional[TrajectoryAugmentation] = None,
    augmentation_after_batch_collate: Optional[TrajectoryAugmentation] = None,
    shuffle: bool = False,
    batch_size: Optional[int] = None
) -> DataLoader:
    """
    Creates dataloader for MOT20 dataset.

    Args:
        cfg: Global config
        split: Split (train/val/test)
        train: Is dataloader created for train or eval
        transform: preprocess-postprocess transform function
        augmentation_before_transform: Augmentations that should be applied BEFORE transform function
        augmentation_after_transform: Augmentations that should be applied AFTER transform function
        augmentation_after_batch_collate: Augmentations that should be applied AFTER batch collate function
        shuffle: Perform shuffle (default: False)
        batch_size: Override config batch size (optional)

    Returns:
        Dataloader for MOT20 dataset.
    """
    assert split in ['train', 'val', 'test'], f'Unknown split "{split}".'

    dataset = TorchTrajectoryDataset(
        dataset=dataset_factory(
            name=cfg.dataset.name,
            path=cfg.dataset.fullpath,
            history_len=cfg.dataset.history_len,
            future_len=cfg.dataset.future_len,
            sequence_list=cfg.dataset.split_index[split],
            additional_params=cfg.dataset.additional_params
        ),
        transform=transform,
        augmentation_before_transform=augmentation_before_transform,
        augmentation_after_transform=augmentation_after_transform
    )

    if batch_size is None:
        batch_size = cfg.train.batch_size if train else cfg.eval.batch_size

    collate_func = OdeDataloaderCollateFunctional(augmentation=augmentation_after_batch_collate)
    return DataLoader(
        dataset=dataset,
        collate_fn=collate_func,
        batch_size=batch_size,
        num_workers=cfg.resources.num_workers,
        shuffle=shuffle
    )


def create_inference_model(
    cfg: GlobalConfig,
    experiment_path: str
) -> nn.Module:
    """
    Creates model from config for inference.

    Args:
        cfg: Config
        experiment_path: Model experiment path

    Returns:
        Initialized model
    """
    checkpoint_path = conventions.get_checkpoint_path(experiment_path, cfg.eval.checkpoint) \
        if cfg.eval.checkpoint else None
    model = load_or_create_model(
        model_type=cfg.model.type,
        params=cfg.model.params,
        checkpoint_path=checkpoint_path
    )
    model = AutoregressiveForecasterDecorator(model, keep_history=cfg.eval.autoregressive_keep_history) \
        if cfg.eval.autoregressive else model
    model.to(cfg.resources.accelerator)
    model.eval()

    return model
