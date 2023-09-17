"""
Implementation of baseline Sort Tracker
"""
import copy
from typing import Optional, Dict, Any, List, Tuple

from nodetracker.library.cv.bbox import PredBBox
from nodetracker.tracker.matching import association_algorithm_factory
from nodetracker.tracker.trackers.base import Tracker
from nodetracker.tracker.tracklet import Tracklet


class BaselineSortTracker(Tracker):
    """
    Baseline Sort tracker components:
    - ReID: HungarianIOU
    - Motion model: Last position
    """
    def __init__(
        self,
        remember_threshold: int,
        matcher_algorithm: str = 'hungarian_iou',
        matcher_params: Optional[Dict[str, Any]] = None,
        use_bbox_id: bool = False
    ):
        """
        Args:
            remember_threshold: How long does the tracklet without any detection matching
                If tracklet isn't matched for more than `remember_threshold` frames then
                it is deleted.
            matcher_algorithm: Choose matching algorithm (e.g. Hungarian IOU)
            matcher_params: Matching algorithm parameters
            use_bbox_id: Use bbox id for tracklet
        """
        matcher_params = {} if matcher_params is None else matcher_params

        # Parameters
        self._remember_threshold = remember_threshold
        self._matcher = association_algorithm_factory(name=matcher_algorithm, params=matcher_params)
        self._use_bbox_id = use_bbox_id

    def track(self, tracklets: List[Tracklet], detections: List[PredBBox], frame_index: int, inplace: bool = True) \
            -> Tuple[List[Tracklet], List[Tracklet], List[Tracklet]]:
        # Note: set `inplace=False` in order to make sure that passed arguments are not changed

        predicted_tracklet_bboxes = [t.bbox for t in tracklets]  # Copy last position
        matches, unmatched_tracklets, unmatched_detections = self._matcher(predicted_tracklet_bboxes, detections)

        # Update matched tracklets history
        for tracklet_index, det_index in matches:
            tracklet = tracklets[tracklet_index] if inplace else copy.deepcopy(tracklets[tracklet_index])
            det_bbox = detections[det_index] if inplace else copy.deepcopy(detections[det_index])
            tracklets[tracklet_index] = tracklet.update(det_bbox, frame_index)

        # Create new tracklets from unmatched detections
        new_tracklets: List[Tracklet] = []
        for det_index in unmatched_detections:
            detection = detections[det_index]
            new_tracklet = Tracklet(
                bbox=detection if inplace else copy.deepcopy(detection),
                frame_index=frame_index
            )
            new_tracklets.append(new_tracklet)

        # Delete old unmatched tracks
        tracklets_indices_to_delete: List[int] = []
        for tracklet_index in unmatched_tracklets:
            if tracklets[tracklet_index].number_of_unmatched_frames(frame_index) > self._remember_threshold:
                tracklets_indices_to_delete.append(tracklet_index)
        deleted_tracklets = [t for i, t in enumerate(tracklets) if i in tracklets_indices_to_delete]
        tracklets = [t for i, t in enumerate(tracklets) if i not in tracklets_indices_to_delete]

        return tracklets, new_tracklets, deleted_tracklets
