"""
Implementation of Filter Sort tracker
"""
import copy
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch

from nodetracker.filter import filter_factory
from nodetracker.library.cv.bbox import PredBBox, BBox
from nodetracker.tracker.matching import association_algorithm_factory
from nodetracker.tracker.trackers.base import Tracker
from nodetracker.tracker.tracklet import Tracklet


class FilterSortTracker(Tracker):
    """
    Baseline Sort tracker components:
    - ReID: HungarianIOU
    - Motion model: Filter prior
    - Combining detection and motion model: Filter posterior
    """
    def __init__(
        self,
        filter_name: str,
        filter_params: dict,
        remember_threshold: int = 1,
        initialization_threshold: int = 3,
        show_only_active: bool = True,
        matcher_algorithm: str = 'hungarian_iou',
        matcher_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            remember_threshold: How long does the tracklet without any detection matching
                If tracklet isn't matched for more than `remember_threshold` frames then
                it is deleted.
            filter_name: Filter name
            filter_params: Filter params
            matcher_algorithm: Choose matching algorithm (e.g. Hungarian IOU)
            matcher_params: Matching algorithm parameters
        """

        # Parameters
        self._remember_threshold = remember_threshold
        self._initialization_threshold = initialization_threshold
        self._show_only_active = show_only_active

        matcher_params = {} if matcher_params is None else matcher_params
        self._matcher = association_algorithm_factory(name=matcher_algorithm, params=matcher_params)

        self._filter = filter_factory(filter_name, filter_params)

        # State
        self._filter_states = {}

    @staticmethod
    def _raw_to_bbox(tracklet: Tracklet, raw: torch.Tensor) -> PredBBox:
        """
        Converts raw tensor to PredBBox for tracked object.

        Args:
            tracklet: Tracklet
            raw: raw bbox coords

        Returns:
            PredBBox
        """
        bbox_raw = raw.numpy().tolist()
        return PredBBox.create(
            bbox=BBox.from_yxwh(*bbox_raw, clip=True),
            label=tracklet.bbox.label,
            conf=tracklet.bbox.conf
        )

    def _initiate(self, tracklet_id: int, detection: PredBBox) -> None:
        """
        Initiates new tracking object state.

        Args:
            tracklet_id: Tracklet id
            detection: Initial object detection
        """
        measurement = torch.from_numpy(detection.as_numpy_yxwh(dtype=np.float32))
        state = self._filter.initiate(measurement)
        self._filter_states[tracklet_id] = state

    def _predict(self, tracklet: Tracklet) -> Tuple[PredBBox, torch.Tensor, torch.Tensor]:
        """
        Estimates object prior position

        Args:
            tracklet: Tracklet (required to fetch state, and bbox info)

        Returns:
            Motion model prior estimation
        """
        assert tracklet.id in self._filter_states, f'Tracklet id "{tracklet.id}" can\'t be found in filter states!'
        state = self._filter_states[tracklet.id]
        state = self._filter.predict(state)
        prior_mean, prior_std = self._filter.project(state)
        self._filter_states[tracklet.id] = state
        bbox = self._raw_to_bbox(tracklet, prior_mean)
        return bbox, prior_mean, prior_std

    def _update(self, tracklet: Tracklet, detection: PredBBox) -> Tuple[PredBBox, torch.Tensor, torch.Tensor]:
        """
        Estimates object posterior position based on the matched detection.

        Args:
            tracklet: Tracklet (required to fetch state, and bbox info)

        Returns:
            Motion model posterior estimation
        """
        measurement = torch.from_numpy(detection.as_numpy_yxwh())

        state = self._filter_states[tracklet.id]
        state = self._filter.update(state, measurement)
        posterior_mean, posterior_std = self._filter.project(state)
        self._filter_states[tracklet.id] = state
        bbox = self._raw_to_bbox(tracklet, posterior_mean)
        return bbox, posterior_mean, posterior_std

    def _missing(self, tracklet: Tracklet) -> Tuple[PredBBox, torch.Tensor, torch.Tensor]:
        """
        Estimates object posterior position for unmatched tracklets (with any detections)

        Args:
            tracklet: Tracklet (required to fetch state, and bbox info)

        Returns:
            Motion model posterior (missing) estimation
        """
        state = self._filter_states[tracklet.id]
        state = self._filter.missing(state)
        posterior_mean, posterior_std = self._filter.project(state)
        self._filter_states[tracklet.id] = state
        bbox = self._raw_to_bbox(tracklet, posterior_mean)
        return bbox, posterior_mean, posterior_std

    def _delete(self, tracklet_id: int) -> None:
        """
        Deletes filter state for deleted tracklet id.

        Args:
            tracklet_id: Tracklet id
        """
        self._filter_states.pop(tracklet_id)

    def track(self, tracklets: List[Tracklet], detections: List[PredBBox], frame_index: int, inplace: bool = True) \
            -> Tuple[List[Tracklet], List[Tracklet]]:
        # Estimate priors for all tracklets
        prior_tracklet_estimates = [self._predict(t) for t in tracklets]  # Copy last position
        prior_tracklet_bboxes = [bbox for bbox, _, _ in prior_tracklet_estimates]  # TODO: Use uncertainty

        # Perform matching
        matches, unmatched_tracklets, unmatched_detections = self._matcher(prior_tracklet_bboxes, detections, tracklets=tracklets)

        # Update matched tracklets
        for tracklet_index, det_index in matches:
            tracklet = tracklets[tracklet_index] if inplace else copy.deepcopy(tracklets[tracklet_index])
            det_bbox = detections[det_index] if inplace else copy.deepcopy(detections[det_index])
            tracklet_bbox, _, _ = self._update(tracklets[tracklet_index], det_bbox)  # TODO: Use uncertainty
            tracklets[tracklet_index] = tracklet.update(tracklet_bbox, frame_index)

        # Create new tracklets from unmatched detections and initiate filter states
        new_tracklets: List[Tracklet] = []
        for det_index in unmatched_detections:
            detection = detections[det_index]
            new_tracklet = Tracklet(
                bbox=detection if inplace else copy.deepcopy(detection),
                frame_index=frame_index
            )
            new_tracklets.append(new_tracklet)
            self._initiate(new_tracklet.id, detection)

        # Delete old unmatched tracks
        tracklets_indices_to_delete: List[int] = []
        for tracklet_index in unmatched_tracklets:
            tracklet = tracklets[tracklet_index]
            if tracklet.number_of_unmatched_frames(frame_index) > self._remember_threshold:
                tracklets_indices_to_delete.append(tracklet_index)
                self._delete(tracklet.id)
            else:
                tracklet_bbox, _, _ = self._missing(tracklet)  # TODO: Use uncertainty
                tracklets[tracklet_index] = tracklet.update(tracklet_bbox, tracklet.frame_index, matched=False)

        all_tracklets = [t for i, t in enumerate(tracklets) if i not in tracklets_indices_to_delete] \
            + new_tracklets

        # Filter active tracklets
        active_tracklets = [t for t in tracklets if t.total_matches >= self._initialization_threshold]
        if self._show_only_active:
            active_tracklets = [t for t in active_tracklets if t.frame_index >= frame_index]

        return active_tracklets, all_tracklets
