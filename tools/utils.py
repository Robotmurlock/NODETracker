"""
Utility functions (often used functions)
"""
from torch.utils.data import DataLoader
from nodetracker.datasets.mot import TorchMOTTrajectoryDataset
from nodetracker.datasets.utils import ode_dataloader_collate_func
from nodetracker.config_parser import GlobalConfig
from nodetracker.datasets.transforms import InvertibleTransform
from typing import Optional


def create_mot20_dataloader(
    dataset_path: str,
    cfg: GlobalConfig,
    train: bool,
    postprocess_transform: Optional[InvertibleTransform] = None,
    shuffle: bool = False,
    batch_size: Optional[int] = None
) -> DataLoader:
    """
    Creates dataloader for MOT20 dataset.

    Args:
        dataset_path: Dataset path
        cfg: Global config
        train: Is dataloader created for train or eval
        postprocess_transform: preprocess-postprocess transform function
        shuffle: Perform shuffle (default: False)
        batch_size: Override config batch size (optional)

    Returns:
        Dataloader for MOT20 dataset.
    """
    dataset = TorchMOTTrajectoryDataset(
        path=dataset_path,
        history_len=cfg.dataset.history_len,
        future_len=cfg.dataset.future_len,
        transform=postprocess_transform
    )

    if batch_size is None:
        batch_size = cfg.train.batch_size if train else cfg.eval.batch_size
    return DataLoader(
        dataset=dataset,
        collate_fn=ode_dataloader_collate_func,
        batch_size=batch_size,
        num_workers=cfg.resources.num_workers,
        shuffle=shuffle
    )
