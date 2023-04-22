"""
Dataset factory
"""
from nodetracker.datasets.torch import TrajectoryDataset
from nodetracker.datasets.mot.core import MOTDataset
from nodetracker.datasets.lasot.core import LaSOTDataset
from typing import Dict, Any, Optional

def dataset_factory(
    name: str,
    path: str,
    history_len: int,
    future_len: int,
    additional_params: Optional[Dict[str, Any]] = None
) -> TrajectoryDataset:
    """
    Creates dataset by given name.

    Args:
        name: Dataset name
        path: Dataset path
        history_len: Observed trajectory length
        future_len: Unobserved trajectory length
        additional_params: Additional dataset parameters

    Returns:
        Initialized dataset
    """
    additional_params = {} if additional_params is None else additional_params

    catalog = {
        'MOT20': MOTDataset,
        'LaSOT': LaSOTDataset
    }

    cls = catalog[name]
    return cls(
        path=path,
        history_len=history_len,
        future_len=future_len,
        **additional_params
    )
