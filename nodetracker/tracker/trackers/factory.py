"""
Tracker factory
"""
from nodetracker.tracker.trackers.base import Tracker
from nodetracker.tracker.trackers.baseline import BaselineSortTracker
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
        'sort-baseline': BaselineSortTracker,
        'filter-sort-tracker': FilterSortTracker
    }

    assert name in TRACKER_CATALOG, f'Unknown tracker "{name}". Available: {list(TRACKER_CATALOG.keys())}'

    return TRACKER_CATALOG[name](**params)