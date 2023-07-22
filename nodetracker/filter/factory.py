from nodetracker.filter.base import StateModelFilter
from nodetracker.filter.kalman_filter import BotSortKalmanFilterWrapper
from nodetracker.filter.node_kalman_filter import NODEKalmanFilter
from nodetracker.filter.node_model_filter import NODEModelFilter, BufferedNodeModelFilter


def filter_factory(name: str, params: dict) -> StateModelFilter:
    """
    Filter (StateModel) factory.

    Args:
        name: Filter name (type)
        params: Filter creation parameters

    Returns:
        Created filter object
    """
    catalog = {
        'node_model': NODEModelFilter,
        'buffered_node_model': BufferedNodeModelFilter,
        'node_kalman': NODEKalmanFilter,
        'akf': BotSortKalmanFilterWrapper
    }

    name = name.lower()
    if name not in catalog:
        raise ValueError(f'Unknown filter "{name}". Available: {list(catalog.keys())}.')

    return catalog[name](**params)
