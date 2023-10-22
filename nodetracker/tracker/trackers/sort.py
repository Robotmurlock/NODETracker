"""
Implementation of Sort Tracker.
"""
import copy
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from nodetracker.library.cv.bbox import PredBBox, BBox
from nodetracker.library.kalman_filter.botsort_kf import BotSortKalmanFilter
from nodetracker.tracker.matching import association_algorithm_factory
from nodetracker.tracker.trackers.base import Tracker
from nodetracker.tracker.tracklet import Tracklet


class SortTracker(Tracker):
    """
    Sort tracker components:
    - ReID: HungarianIOU
    - Motion model: Kalman filter
    """
    def __init__(
        self,
        remember_threshold: int = 1,
        initialization_threshold: int = 3,
        new_tracklet_detection_threshold: Optional[float] = None,
        show_only_active: bool = True,
        matcher_algorithm: str = 'hungarian_iou',
        matcher_params: Optional[Dict[str, Any]] = None,
        no_motion: bool = False,
        kf_args: Optional[dict] = None
    ):
        """
        Args:
            remember_threshold: How long is the tracklet remembered without any detection matching
                If tracklet isn't matched for more than `remember_threshold` frames then
                it is deleted.
            initialization_threshold: Number of matched required to make
            show_only_active: Show only active tracklets (matched in this frame)
            matcher_algorithm: Choose matching algorithm (e.g. Hungarian IOU)
            matcher_params: Matching algorithm parameters
            no_motion: Optionally disable KF motion model
            kf_args: Custom configuration for the KF
        """
        # Parameters
        self._remember_threshold = remember_threshold
        self._initialization_threshold = initialization_threshold
        self._show_only_active = show_only_active
        self._new_tracklet_detection_threshold = new_tracklet_detection_threshold

        matcher_params = {} if matcher_params is None else matcher_params
        self._matcher = association_algorithm_factory(name=matcher_algorithm, params=matcher_params)
        self._no_motion = no_motion

        if kf_args is None:
            kf_args = {}
        self._kf = BotSortKalmanFilter(**kf_args)

        # State - KF
        self._kf_states: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _raw_to_bbox(tracklet: Tracklet, raw: np.ndarray) -> PredBBox:
        """
        Converts raw tensor to PredBBox for tracked object.

        Args:
            tracklet: Tracklet
            raw: raw bbox coords

        Returns:
            PredBBox
        """
        bbox_raw = raw.tolist()
        return PredBBox.create(
            bbox=BBox.from_yxwh(*bbox_raw, clip=True),
            label=tracklet.bbox.label,
            conf=tracklet.bbox.conf
        )

    def _initiate(self, tracklet_id: int, detection: PredBBox) -> None:
        """
        Initiates tracklet KF state.

        Args:
            tracklet_id: Tracklet id
            detection: New detection
        """
        if self._no_motion:
            return

        measurement = detection.as_numpy_yxwh()
        mean, cov = self._kf.initiate(measurement)
        self._kf_states[tracklet_id] = (mean, cov)

    def _predict(self, tracklet: Tracklet) -> PredBBox:
        """
        Predicts tracklet position in the new frame

        Args:
            tracklet: Object tracklet

        Returns:
            Updated bbox
        """
        if self._no_motion:
            return tracklet.bbox

        assert tracklet.id in self._kf_states, f'Tracklet id "{tracklet.id}" can\'t be found in filter states!'
        mean, cov = self._kf_states[tracklet.id]
        z_prior_mean, z_prior_cov = self._kf.predict(mean, cov)
        prior_mean, _ = self._kf.project(z_prior_mean, z_prior_cov)
        bbox = self._raw_to_bbox(tracklet, prior_mean)
        self._kf_states[tracklet.id] = (z_prior_mean, z_prior_cov)
        return bbox

    def _update(self, tracklet: Tracklet, detection: PredBBox) -> PredBBox:
        """
        Updates tracklet position in case of successful match with some detection in the current frame.

        Args:
            tracklet: Object tracklet
            detection: Object bbox

        Returns:
            Updated bbox
        """
        if self._no_motion:
            return tracklet.bbox

        measurement = detection.as_numpy_yxwh(dtype=np.float32)

        mean, cov = self._kf_states[tracklet.id]
        z_posterior_mean, z_posterior_cov = self._kf.update(mean, cov, measurement)
        posterior_mean, _ = self._kf.project(z_posterior_mean, z_posterior_cov)
        bbox = self._raw_to_bbox(tracklet, posterior_mean)
        self._kf_states[tracklet.id] = (z_posterior_mean, z_posterior_cov)
        return bbox

    def _missing(self, tracklet: Tracklet) -> PredBBox:
        """
        Estimated tracklet position in case of failed match with any detections in the current frame.

        Args:
            tracklet: Object tracklet

        Returns:
            Updated bbox
        """
        z_mean, z_cov = self._kf_states[tracklet.id]
        mean, _ = self._kf.project(z_mean, z_cov)
        return self._raw_to_bbox(tracklet, mean)

    def _delete(self, tracklet_id: int) -> None:
        """
        Deletes tracklet of a disappeared object.

        Args:
            tracklet_id: Tracklet id
        """
        self._kf_states.pop(tracklet_id)

    def track(self, tracklets: List[Tracklet], detections: List[PredBBox], frame_index: int, inplace: bool = True) \
            -> Tuple[List[Tracklet], List[Tracklet]]:
        # Motion estimation
        predicted_tracklet_bboxes = [self._predict(t) for t in tracklets]  # Copy last position

        # Matching tracklets with detections
        matches, unmatched_tracklets, unmatched_detections = self._matcher(predicted_tracklet_bboxes, detections, tracklets=tracklets)

        # Update matched tracklets history
        for tracklet_index, det_index in matches:
            tracklet = tracklets[tracklet_index] if inplace else copy.deepcopy(tracklets[tracklet_index])
            det_bbox = detections[det_index] if inplace else copy.deepcopy(detections[det_index])
            tracklet_bbox = self._update(tracklets[tracklet_index], det_bbox)
            tracklets[tracklet_index] = tracklet.update(tracklet_bbox, frame_index, matched=True)

        # Create new tracklets from unmatched detections
        new_tracklets: List[Tracklet] = []
        for det_index in unmatched_detections:
            detection = detections[det_index]
            if self._new_tracklet_detection_threshold is not None and detection.conf < self._new_tracklet_detection_threshold:
                continue

            new_tracklet = Tracklet(
                bbox=detection if inplace else copy.deepcopy(detection),
                frame_index=frame_index
            )
            self._initiate(new_tracklet.id, detection)
            new_tracklets.append(new_tracklet)

        # Delete old unmatched tracks
        tracklets_indices_to_delete: List[int] = []
        for tracklet_index in unmatched_tracklets:
            tracklet = tracklets[tracklet_index]
            if tracklet.number_of_unmatched_frames(frame_index) > self._remember_threshold:
                tracklets_indices_to_delete.append(tracklet_index)
                self._delete(tracklet.id)
            else:
                tracklet_bbox = self._missing(tracklet)
                tracklets[tracklet_index] = tracklet.update(tracklet_bbox, tracklet.frame_index, matched=False)

        all_tracklets = [t for i, t in enumerate(tracklets) if i not in tracklets_indices_to_delete] \
            + new_tracklets

        # Filter active tracklets
        active_tracklets = [t for t in tracklets if t.total_matches >= self._initialization_threshold]
        if self._show_only_active:
            active_tracklets = [t for t in active_tracklets if t.frame_index >= frame_index]

        return active_tracklets, all_tracklets
