"""
Tracker factory method.
"""
from nodetracker.tracker.trackers.base import Tracker
from nodetracker.tracker.trackers.sort import SortTracker
from nodetracker.tracker.trackers.filter_sort import FilterSortTracker

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
        'filter-sort-tracker': FilterSortTracker
    }

    assert name in TRACKER_CATALOG, f'Unknown tracker "{name}". Available: {list(TRACKER_CATALOG.keys())}'

    return TRACKER_CATALOG[name](**params)