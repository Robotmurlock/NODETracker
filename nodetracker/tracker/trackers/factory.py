"""
Tracker factory method.
"""
from nodetracker.tracker.trackers.base import Tracker
from nodetracker.tracker.trackers.byte import ByteTracker
from nodetracker.tracker.trackers.sort import SortTracker
from nodetracker.tracker.trackers.sparse import SparseTracker


def tracker_factory(name: str, params: dict) -> Tracker:
    """
    Tracker factory

    Args:
        name: tracker name
        params: tracker params

    Returns:
        Tracker object
    """
    name = name.lower()

    TRACKER_CATALOG = {
        'sort': SortTracker,
        'filter-sort-tracker': SortTracker,  # alias
        'byte': ByteTracker,
        'sparse': SparseTracker
    }

    assert name in TRACKER_CATALOG, f'Unknown tracker "{name}". Available: {list(TRACKER_CATALOG.keys())}'

    return TRACKER_CATALOG[name](**params)