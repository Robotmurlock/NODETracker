"""
Tracker. Supports:
- SORT
- Association algorithms (Hungarian)
"""
from nodetracker.tracker.matching import AssociationAlgorithm, HungarianAlgorithmIOU
from nodetracker.tracker.trackers.base import Tracker
from nodetracker.tracker.trackers.baseline import BaselineSortTracker
from nodetracker.tracker.trackers.factory import tracker_factory
from nodetracker.tracker.tracklet import Tracklet
