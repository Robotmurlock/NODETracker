"""
Implementation of ByteTrack.
"""
import copy
from typing import Optional, Dict, Any, List, Tuple

from nodetracker.library.cv.bbox import PredBBox
from nodetracker.tracker.matching import association_algorithm_factory
from nodetracker.tracker.trackers.motion import MotionBasedTracker
from nodetracker.tracker.tracklet import Tracklet, TrackletState
from nodetracker.utils.collections import unpack_n
from nodetracker.tracker.trackers.utils import remove_duplicates


class ByteTracker(MotionBasedTracker):
    """
    ByteTrack algorithm.

    Steps:
        0. Estimate all tracklets priors
        1. Split detections into high and low
        2. Match high detections with tracklets with states ACTIVE and LOST using HighMatchAlgorithm
        3. Match remaining ACTIVE tracklets with low detections using LowMatchAlgorithm
        4. Mark remaining (unmatched) tracklets as lost
        5. Match NEW tracklets with high detections using NewMatchAlgorithm
            - remove all NEW unmatched tracklets
        6. Initialize new tracklets from unmatched high detections
        7. Update matched tracklets
        8. Delete new unmatched and long-lost tracklets
        9. Delete duplicate between ACTIVE and LOST tracklets
    """
    def __init__(
        self,
        filter_name: str,
        filter_params: dict,
        high_matcher_algorithm: str = 'default',
        high_matcher_params: Optional[Dict[str, Any]] = None,
        low_matcher_algorithm: str = 'default',
        low_matcher_params: Optional[Dict[str, Any]] = None,
        new_matcher_algorithm: str = 'default',
        new_matcher_params: Optional[Dict[str, Any]] = None,
        detection_threshold: float = 0.6,
        remember_threshold: int = 1,
        initialization_threshold: int = 3,
        new_tracklet_detection_threshold: Optional[float] = None,
        duplicate_iou_threshold: float = 0.85,
        use_observation_if_lost: bool = False
    ):
        super().__init__(
            filter_name=filter_name,
            filter_params=filter_params
        )

        if high_matcher_algorithm == 'default':
            assert high_matcher_params is None
            high_matcher_algorithm = 'hungarian_iou'
            high_matcher_params = {
                'match_threshold': 0.2,
                'fuse_score': True
            }

        self._high_match = association_algorithm_factory(high_matcher_algorithm, high_matcher_params)

        if low_matcher_algorithm == 'default':
            assert low_matcher_params is None
            low_matcher_algorithm = 'hungarian_iou'
            low_matcher_params = {
                'match_threshold': 0.5
            }

        self._low_match = association_algorithm_factory(low_matcher_algorithm, low_matcher_params)

        if new_matcher_algorithm == 'default':
            assert new_matcher_params is None
            new_matcher_algorithm = 'hungarian_iou'
            new_matcher_params = {
                'match_threshold': 0.3,
                'fuse_score': True
            }

        self._new_match = association_algorithm_factory(new_matcher_algorithm, new_matcher_params)

        # Parameters
        self._initialization_threshold = initialization_threshold
        self._remember_threshold = remember_threshold
        self._use_observation_if_lost = use_observation_if_lost
        self._detection_threshold = detection_threshold
        self._new_tracklet_detection_threshold = new_tracklet_detection_threshold \
            if new_tracklet_detection_threshold is not None else detection_threshold
        self._duplicate_iou_threshold = duplicate_iou_threshold

        # State
        self._next_id = 0


    def track(self, tracklets: List[Tracklet], detections: List[PredBBox], frame_index: int, inplace: bool = True) \
            -> Tuple[List[Tracklet], List[Tracklet]]:
        # (0) Estimate priors for all tracklets
        prior_tracklet_estimates = [self._predict(t) for t in tracklets]
        prior_tracklet_bboxes = [bbox for bbox, _, _ in prior_tracklet_estimates]

        # (1) Split detections into low and high
        high_detections = [d for d in detections if d.conf >= self._detection_threshold]
        high_det_indices = [i for i, d in enumerate(detections) if d.conf >= self._detection_threshold]
        low_detections = [d for d in detections if d.conf < self._detection_threshold]
        low_det_indices = [i for i, d in enumerate(detections) if d.conf < self._detection_threshold]

        # (2) Match high detections with tracklets with states ACTIVE and LOST using HighMatchAlgorithm
        tracklets_active_and_lost_indices, tracklets_active_and_lost, tracklets_active_and_lost_bboxes = \
            unpack_n([(i, t, t_bbox) for i, (t, t_bbox) in enumerate(zip(tracklets, prior_tracklet_bboxes)) if t.is_tracked], n=3)
        high_matches, remaining_tracklet_indices, high_unmatched_detections_indices = \
            self._high_match(tracklets_active_and_lost_bboxes, high_detections, tracklets_active_and_lost)
        high_matches = [(tracklets_active_and_lost_indices[t_i], high_det_indices[d_i]) for t_i, d_i in high_matches]
        high_unmatched_detections_indices = [high_det_indices[d_i] for d_i in high_unmatched_detections_indices]
        remaining_tracklet_bboxes = [tracklets_active_and_lost_bboxes[t_i] for t_i in remaining_tracklet_indices]
        remaining_tracklets = [tracklets_active_and_lost[t_i] for t_i in remaining_tracklet_indices]
        remaining_tracklet_indices = [tracklets_active_and_lost_indices[t_i] for t_i in remaining_tracklet_indices]

        # (3) Match remaining ACTIVE tracklets with low detections using LowMatchAlgorithm
        remaining_active_tracklet_indices, remaining_active_tracklets, remaining_active_tracklet_bboxes = \
            unpack_n([(i, t, t_bbox) for i, t, t_bbox in zip(remaining_tracklet_indices, remaining_tracklets, remaining_tracklet_bboxes)
                  if t.state == TrackletState.ACTIVE], n=3)
        low_matches, low_unmatched_tracklet_indices, _ = \
            self._low_match(remaining_active_tracklet_bboxes, low_detections, remaining_active_tracklets)
        low_matches = [(remaining_active_tracklet_indices[t_i], low_det_indices[d_i]) for t_i, d_i in low_matches]
        unmatched_tracklet_indices = [remaining_active_tracklet_indices[t_i] for t_i in low_unmatched_tracklet_indices]

        # (4) Mark remaining (unmatched) tracklets as lost
        for t_i in unmatched_tracklet_indices:
            tracklets[t_i].state = TrackletState.LOST

        # (5) Match NEW tracklets with high detections using NewMatchAlgorithm
        remaining_high_detections = [detections[d_i] for d_i in high_unmatched_detections_indices]
        remaining_high_detection_indices = high_unmatched_detections_indices
        tracklets_new_indices, tracklets_new, tracklets_new_bboxes = \
            unpack_n([(i, t, t_bbox) for i, (t, t_bbox) in enumerate(zip(tracklets, prior_tracklet_bboxes)) if t.state == TrackletState.NEW], n=3)
        new_matches, _, new_unmatched_detections_indices = \
            self._new_match(tracklets_new_bboxes, remaining_high_detections, tracklets_new)
        new_matches = [(tracklets_new_indices[t_i], high_unmatched_detections_indices[d_i]) for t_i, d_i in new_matches]
        new_unmatched_detections_indices = [remaining_high_detection_indices[d_i] for d_i in new_unmatched_detections_indices]

        # (6) Initialize new tracklets from unmatched high detections
        new_tracklets: List[Tracklet] = []
        for d_i in new_unmatched_detections_indices:
            detection = detections[d_i]
            if self._new_tracklet_detection_threshold is not None and detection.conf < self._new_tracklet_detection_threshold:
                continue

            new_tracklet = Tracklet(
                bbox=detection if inplace else copy.deepcopy(detection),
                frame_index=frame_index,
                _id=self._next_id,
                state=TrackletState.NEW if frame_index > 1 else TrackletState.ACTIVE
            )
            self._next_id += 1
            new_tracklets.append(new_tracklet)
            self._initiate(new_tracklet.id, detection)

        all_matches = high_matches + low_matches + new_matches

        # (7) Update matched tracklets
        for tracklet_index, det_index in all_matches:
            tracklet = tracklets[tracklet_index] if inplace else copy.deepcopy(tracklets[tracklet_index])
            det_bbox = detections[det_index] if inplace else copy.deepcopy(detections[det_index])
            tracklet_bbox, _, _ = self._update(tracklets[tracklet_index], det_bbox)
            new_bbox = det_bbox if self._use_observation_if_lost and tracklet.state != TrackletState.ACTIVE \
                else tracklet_bbox
            tracklets[tracklet_index] = tracklet.update(new_bbox, frame_index, state=TrackletState.ACTIVE)

        # (8) Delete new unmatched and long-lost tracklets
        tracklets_indices_to_delete: List[int] = []
        for tracklet_index in range(len(tracklets)):
            tracklet = tracklets[tracklet_index]
            if tracklet.state == TrackletState.ACTIVE:
                continue

            if (tracklet.state == TrackletState.LOST
                and tracklet.number_of_unmatched_frames(frame_index) > self._remember_threshold) \
                    or tracklet.state == TrackletState.NEW:
                tracklets_indices_to_delete.append(tracklet_index)
                self._delete(tracklet.id)
            else:
                tracklet_bbox, _, _ = self._missing(tracklet)
                tracklets[tracklet_index] = tracklet.update(tracklet_bbox, tracklet.frame_index, state=TrackletState.LOST)

        # (9) Delete duplicate between ACTIVE and LOST tracklets
        # all_tracklets = [t for i, t in enumerate(tracklets) if i not in tracklets_indices_to_delete] + new_tracklets
        filtered_tracklets = [t for i, t in enumerate(tracklets) if i not in tracklets_indices_to_delete]
        active_tracklets = [t for t in filtered_tracklets if t.state == TrackletState.ACTIVE]
        lost_tracklets = [t for t in filtered_tracklets if t.state == TrackletState.LOST]
        active_tracklets, lost_tracklets = remove_duplicates(self._duplicate_iou_threshold, active_tracklets, lost_tracklets)
        all_tracklets = active_tracklets + lost_tracklets + new_tracklets

        # Filter active tracklets
        active_tracklets = [t for t in tracklets if t.total_matches >= self._initialization_threshold]
        active_tracklets = [t for t in active_tracklets if t.state == TrackletState.ACTIVE]

        return active_tracklets, all_tracklets
