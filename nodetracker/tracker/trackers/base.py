"""

"""
from abc import ABC, abstractmethod
from typing import List, Tuple

from nodetracker.library.cv.bbox import PredBBox
from nodetracker.tracker.tracklet import Tracklet


class Tracker(ABC):
    """
    Tracker interface
    """
    @abstractmethod
    def track(
        self,
        tracklets: List[Tracklet],
        detections: List[PredBBox],
        frame_index: int,
        inplace: bool = True
    ) -> Tuple[List[Tracklet], List[Tracklet], List[int]]:
        """
        Performs multi-object-tracking step.
        DISCLAIMER: `inplace=True` is default configuration!

        Args:
            tracklets: List of active trackelts
            detections: Lists of new detections
            frame_index: Current frame number
            inplace: Perform inplace transformations on tracklets and bboxes

        Returns:
            - List of new trackelts
            - List of updated tracklets (existing)
            - List of deleted tracklets
        """
        pass
