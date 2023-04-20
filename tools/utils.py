"""
Utility functions (often used functions)
"""
from typing import Optional

from torch.utils.data import DataLoader

from nodetracker.config_parser import GlobalConfig
from nodetracker.datasets import TorchTrajectoryDataset, dataset_factory
from nodetracker.datasets.augmentations import TrajectoryAugmentation
from nodetracker.datasets.transforms import InvertibleTransform
from nodetracker.datasets.utils import OdeDataloaderCollateFunctional


def create_dataloader(
    dataset_path: str,
    cfg: GlobalConfig,
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
        dataset_path: Dataset path
        cfg: Global config
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
    dataset = TorchTrajectoryDataset(
        dataset=dataset_factory(
            name=cfg.dataset.name,
            path=dataset_path,
            history_len=cfg.dataset.history_len,
            future_len=cfg.dataset.future_len,
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
