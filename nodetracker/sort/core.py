"""
Implementation of SORT algorithm variation
https://arxiv.org/pdf/1602.00763.pdf
"""
from typing import List

import torch
from copy import deepcopy

from nodetracker.library.cv import PredBBox, BBox, Point
from nodetracker.node import BBoxTrajectoryForecaster
from nodetracker.sort.matching import AssociationAlgorithm


def forecast_to_bboxes(forecast: torch.Tensor) -> List[PredBBox]:
    """
    Converts raw forecast output to list of PredBBox items.

    Args:
        forecast: Forecasts

    Returns:
        Pred bboxes.
    """
    shape = forecast.shape
    assert len(shape) == 3, f'Expected shape length to be 3. Found {shape}.'
    assert shape[1] == 1, f'Currently supporting only batch size 1. Found {shape}.'
    forecast = forecast[:, 0, :]

    bboxes: List[PredBBox] = []
    for f_i in range(forecast.shape[0]):
        raw_bbox = forecast[f_i].numpy()
        bbox = PredBBox.create(BBox.from_xyhw(*raw_bbox, clip=True), label=0)
        bboxes.append(bbox)

    return bboxes

class SortTracker:
    """
    Algorithm:
        State:
        - Active objects: tracked objects with matched detections in previous frame
        - Lost objects: tracked objects that do not have matched detection in last few frames
        - New objects: New objects

        Steps:
        1. Use forecaster (e.g. KalmanFilter) to predict bbox coords of active and lost objects
        2. Match
    """
    def __init__(
        self,
        forecaster: BBoxTrajectoryForecaster,
        matcher: AssociationAlgorithm,
        object_persist_frames: int = 5
    ):
        """
        Args:
            forecaster: Forecasts bbox coords of tracked object in next frame
            matcher: Tracklet-detection matching
            object_persist_frames: Number of frames to track lost objects before removing them
        """
        # Algorithms
        self._forecaster = forecaster
        self._matcher = matcher

        # Active tracklets state (matched with some detections in last step)
        self._active_tracklets: List[PredBBox] = []
        self._active_tracklets_features: List[torch.Tensor] = []
        self._active_tracklets_forecasts: List[List[PredBBox]] = []

        # Missing tracklets state (tracklets that are missing detections)
        self._missing_tracklets: List[PredBBox] = []
        self._missing_tracklets_features: List[torch.Tensor] = []
        self._missing_tracklets_counter: List[int] = []
        self._missing_tracklets_forecasts: List[List[PredBBox]] = []

        # Hyperparameters
        self._object_persist_frames = object_persist_frames

        # State
        self._next_object_index = 0

    def _create_next_object_index(self) -> int:
        object_index = self._next_object_index
        self._next_object_index += 1
        return object_index

    def _predict_active(self, features: List[torch.Tensor], forecasts: List[List[PredBBox]]) -> List[PredBBox]:
        # TODO: Currently not using batch predictions
        estimated_tracklet_coords: List[PredBBox] = []

        for tracklet_index, feats in enumerate(features):
            t_obs = 1 + torch.tensor(list(range(feats.shape[0])), dtype=torch.float32).view(-1, 1, 1)
            t_unobs = (t_obs.shape[0] + 1) + torch.tensor(list(range(self._object_persist_frames)), dtype=torch.float32).view(-1, 1, 1)
            forecast, *_ = self._forecaster(feats, t_obs, t_unobs)
            forecast_bboxes = forecast_to_bboxes(forecast)
            forecasts[tracklet_index] = forecast_bboxes
            estimated_tracklet_coords.append(forecast_bboxes[0])

        return estimated_tracklet_coords

    @staticmethod
    def _predict_missing(forecasts: List[List[PredBBox]], counter: List[int]) -> List[PredBBox]:
        estimated_tracklet_coords: List[PredBBox] = []

        for forecast, cnt in zip(forecasts, counter):
            bbox = forecast[cnt]
            estimated_tracklet_coords.append(bbox)

        return estimated_tracklet_coords

    def _move_tracklets_to_active(self, matched_missing_tracklets: List[int]) -> None:
        # Removing from last to first index doesn't change position of next tracklet that is moved
        matched_missing_tracklets = sorted(matched_missing_tracklets, reverse=True)
        for t_i in matched_missing_tracklets:
            # Remove from missing
            tracklet = self._missing_tracklets.pop(t_i)
            features = self._missing_tracklets_features.pop(t_i)
            forecasts = self._missing_tracklets_forecasts.pop(t_i)
            self._missing_tracklets_counter.pop(t_i)

            # Add to active
            self._active_tracklets.append(tracklet)
            self._active_tracklets_features.append(features)
            self._active_tracklets_forecasts.append(forecasts)

    def _move_tracklets_to_missing(self, unmatched_active_tracklets: List[int]) -> None:
        # Removing from last to first index doesn't change position of next tracklet that is moved
        unmatched_active_tracklets = sorted(unmatched_active_tracklets, reverse=True)
        for t_i in unmatched_active_tracklets:
            # Remove from active
            tracklet = self._active_tracklets.pop(t_i)
            features = self._active_tracklets_features.pop(t_i)
            forecasts = self._active_tracklets_forecasts.pop(t_i)

            # Add to missing
            self._missing_tracklets.append(tracklet)
            self._missing_tracklets_features.append(features)
            self._missing_tracklets_forecasts.append(forecasts)
            self._missing_tracklets_counter.append(1)

    def _create_new_tracklets(self, detections: List[PredBBox]) -> None:
        for d in detections:
            d.label = self._create_next_object_index()
            self._active_tracklets.append(d)
            self._active_tracklets_forecasts.append([d])

            bbox_array = d.as_numpy_xyhw()
            initial_features = torch.from_numpy(bbox_array).view(1, 1, 4)
            self._active_tracklets_features.append(initial_features)

    def track(self, detections: List[PredBBox]) -> List[PredBBox]:
        """
        Updates tracklets with new detections

        Args:
            detections: List of new detected objects\\\

        Returns:
            List of tracklets for active and missing objects
        """
        # Match detections with active tracklets
        estimated_active_tracklets_bboxes = self._predict_active(self._active_tracklets_features, self._active_tracklets_forecasts)
        active_matches, unmatched_tracklets_indices, unmatched_detections_indices = self._matcher(estimated_active_tracklets_bboxes, detections)
        # TODO: Update features for active matches
        # TODO: Update coords for active matches

        # Match remaining detections with missing tracklets
        remaining_detections = [detections[ud_index] for ud_index in unmatched_detections_indices]
        estimated_missing_tracklets_bboxes = self._predict_missing(self._missing_tracklets_forecasts, self._missing_tracklets_counter)
        missing_matches, _, unmatched_detections_indices = self._matcher(estimated_missing_tracklets_bboxes, remaining_detections)
        # TODO: Update coords for missing matches (features should not be updated)

        # Move matched missing tracklets to active tracklets
        matched_missing_tracklets = [t_i for t_i, d_i in missing_matches]
        self._move_tracklets_to_active(matched_missing_tracklets)

        # Move unmatched active tracklets to missing tracklets
        self._move_tracklets_to_missing(unmatched_tracklets_indices)
        unmatched_detections = [detections[ud_index] for ud_index in unmatched_detections_indices]

        # Add unmatched detections to active tracklets
        # TODO: New objects should now appear immediately? FP
        # TODO: Update features for new objects
        self._create_new_tracklets(unmatched_detections)

        return deepcopy(self._active_tracklets)


def run_test() -> None:
    from nodetracker.node.kalman_filter import TorchConstantVelocityODKalmanFilter
    from nodetracker.sort.matching import HungarianAlgorithmIOU

    forecaster = TorchConstantVelocityODKalmanFilter()
    matcher = HungarianAlgorithmIOU(match_threshold=0.2)
    tracker = SortTracker(forecaster, matcher)

    fst_iter_detections = [
        PredBBox(
            label=0,
            conf=0.9,
            upper_left=Point(x=0.1, y=0.1),
            bottom_right=Point(x=0.2, y=0.2)
        ),
        PredBBox(
            label=0,
            conf=0.9,
            upper_left=Point(x=0.5, y=0.5),
            bottom_right=Point(x=0.7, y=0.7)
        )
    ]

    snd_iter_detections = [
        PredBBox(
            label=0,
            conf=0.9,
            upper_left=Point(x=0.11, y=0.11),
            bottom_right=Point(x=0.21, y=0.21)
        ),
        PredBBox(
            label=0,
            conf=0.9,
            upper_left=Point(x=0.51, y=0.51),
            bottom_right=Point(x=0.71, y=0.71)
        )
    ]

    print(tracker.track(fst_iter_detections))
    print(tracker.track(snd_iter_detections))


if __name__ == '__main__':
    run_test()
