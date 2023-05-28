"""
TODO: REFACTOR

Implementation of SORT algorithm variation
https://arxiv.org/pdf/1602.00763.pdf

TODO: Refactor: Too many states (self...) :'(
"""
import logging
from copy import deepcopy
from typing import List, Tuple, Optional

import torch

from nodetracker.datasets import transforms
from nodetracker.library.cv import PredBBox, BBox
from nodetracker.node import BBoxTrajectoryForecaster
from nodetracker.sort.matching import AssociationAlgorithm

logger = logging.getLogger('SortTracker')

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
        tensor_transform: Optional[transforms.InvertibleTransform] = None,
        accelerator: str = 'cpu',
        object_persist_frames: int = 5,
        max_feature_length: int = 15
    ):
        """
        Args:
            forecaster: Forecasts bbox coords of tracked object in next frame
            matcher: Tracklet-detection matching
            tensor_transform: Forecaster data preprocess and postprocess
            accelerator: Forecaster accelerator (gpu/cpu)
            object_persist_frames: Number of frames to track lost objects before removing them
            max_feature_length: Max feature length before popping history
        """
        # Algorithms
        self._forecaster= forecaster
        self._matcher = matcher
        self._tensor_transform = tensor_transform
        if self._tensor_transform is None:
            self._tensor_transform = transforms.IdentityTransform()
        self._accelerator = accelerator
        # noinspection PyUnresolvedReferences
        self._forecaster.to(self._accelerator)

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
        self._max_feature_length = max_feature_length

        # State
        self._next_object_index = 0
        self._iter_index = 1

    @staticmethod
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

    def _create_next_object_index(self) -> int:
        """
        Creates new tracklet id.

        Returns:
            New tracklet id
        """
        object_index = self._next_object_index
        self._next_object_index += 1
        return object_index

    def _predict_active(self) -> List[PredBBox]:
        """
        Estimates tracklet bbox coordinates in next N steps.
        Only next step estimations are returned and other values are saved as `forecasts`.

        Returns:
            Current frame estimated bbox locations for active tracklets.
        """
        # TODO: Currently not using batch predictions
        estimated_tracklet_coords: List[PredBBox] = []

        for tracklet_index, feats in enumerate(self._active_tracklets_features):
            # Initialize t_obs/t_unobs time points
            # t_obs and t_unobs example: [1, 2, 3, 4], [5, 6, 7, 8]
            n_obs = feats.shape[0]
            n_unobs = self._object_persist_frames
            t_obs = torch.linspace(1, n_obs, steps=n_obs, dtype=torch.float32).view(-1, 1, 1)
            t_unobs = torch.linspace(n_obs+1, n_obs+n_unobs, steps=n_unobs, dtype=torch.float32).view(-1, 1, 1)

            if feats.shape[0] > 1:
                f_feats, _, f_t_obs, f_t_unobs = self._tensor_transform.apply([feats, None, t_obs, t_unobs], shallow=False)
                f_feats, f_t_obs, f_t_unobs = [val.to(self._accelerator) for val in [f_feats, f_t_obs, f_t_unobs]]
                forecast, *_ = self._forecaster(f_feats, f_t_obs, f_t_unobs)
                _, forecast, *_ = self._tensor_transform.inverse([feats, forecast, f_t_obs, f_t_unobs])
                forecast = forecast.detach().cpu()
                forecast_bboxes = self.forecast_to_bboxes(forecast)
            else:
                # Not enough data for forecast (replicating same bbox)
                forecast_bboxes = [self._active_tracklets[tracklet_index]]

            estimated_tracklet_coords.append(forecast_bboxes[0])  # only next step forecast estimation is required for matching
            self._active_tracklets_forecasts[tracklet_index] = forecast_bboxes  # save forecasts

        return estimated_tracklet_coords

    def _update_active_tracklets(self, matched_active_tracklet_indices: List[int], matched_active_detections: List[PredBBox]) -> None:
        """
        For each active tracklet performs:
        - Update tracklet bbox
        - Update tracklet features

        Args:
            matched_active_tracklet_indices: tracklet indices that are matched with some detected object in current frame
            matched_active_detections: bboxes of matched detected objects
        """
        for t_i, bbox in zip(matched_active_tracklet_indices, matched_active_detections):
            # TODO: Currently using naive update approach - take detection bbox - instead of combining it like in KF
            self._active_tracklets[t_i] = PredBBox.create(bbox, label=self._active_tracklets[t_i].label, conf=self._active_tracklets[t_i].conf)
            features = self._active_tracklets_features[t_i][:self._max_feature_length]
            bbox_features = torch.from_numpy(bbox.as_numpy_xyhw()).view(1, 1, 4)  # TODO: batch size is fixed to 1
            features = torch.cat([features, bbox_features], dim=0)  # TODO: Might not be most efficient way...
            self._active_tracklets_features[t_i] = features

    def _predict_missing(self) -> List[PredBBox]:
        """
        "Estimates" tracklet bbox coordinates.
        Takes bbox from saved forecasts based on missing frame counter.

        Returns:
            Current frame estimated bbox locations for missing tracklets.
        """
        estimated_tracklet_coords: List[PredBBox] = []

        for forecast, cnt in zip(self._missing_tracklets_forecasts, self._missing_tracklets_counter):
            bbox = forecast[cnt]
            estimated_tracklet_coords.append(bbox)

        return estimated_tracklet_coords

    def _move_tracklets_to_active(self, tracklet_indices: List[int], matched_detections: List[PredBBox]) -> None:
        """
        Moves matched missing tracklets to active group.

        Args:
            tracklet_indices: List of previously missing tracklets indices that are matched again.
            matched_detections: List of matched detections
        """
        # Removing from last to first index doesn't change position of next tracklet that is moved
        tracklet_indices = sorted(tracklet_indices, reverse=True)
        for t_i, bbox in zip(tracklet_indices, matched_detections):
            tracklet_id = self._missing_tracklets[t_i].label
            logger.debug(f'Moving tracklet {tracklet_id} from missing to active.')

            # Remove from missing
            tracklet = self._missing_tracklets.pop(t_i)
            features = self._missing_tracklets_features.pop(t_i)
            forecasts = self._missing_tracklets_forecasts.pop(t_i)
            self._missing_tracklets_counter.pop(t_i)

            # Update tracklet bbox
            tracklet = PredBBox.create(bbox=bbox, label=tracklet.label, conf=tracklet.conf)
            # TODO: Currently using naive update approach

            # Add to active
            self._active_tracklets.append(tracklet)
            self._active_tracklets_features.append(features)
            self._active_tracklets_forecasts.append(forecasts)

    def _move_tracklets_to_missing(self, unmatched_active_tracklets: List[int]) -> None:
        """
        Moves unmatched active tracklets to missing group.

        Args:
            unmatched_active_tracklets: List of active tracklets that are were not matched in current frame.
        """
        # Removing from last to first index doesn't change position of next tracklet that is moved
        unmatched_active_tracklets = sorted(unmatched_active_tracklets, reverse=True)
        for t_i in unmatched_active_tracklets:
            tracklet_id = self._active_tracklets[t_i].label
            logger.debug(f'Moving tracklet {tracklet_id} from active to missing.')

            # Remove from active
            tracklet = self._active_tracklets.pop(t_i)
            features = self._active_tracklets_features.pop(t_i)
            forecasts = self._active_tracklets_forecasts.pop(t_i)

            # Add to missing
            self._missing_tracklets.append(tracklet)
            self._missing_tracklets_features.append(features)
            self._missing_tracklets_forecasts.append(forecasts)
            self._missing_tracklets_counter.append(0)

    def _update_missing_tracklets(self) -> None:
        """
        For each tracklet:
        - if tracklet is missing more than `object_persist_frames` then it is removed from missing tracklets list
        - otherwise:
            - missing frames counter is increased (this value is also used to choose forecast bbox)
        """
        keep_indices: List[int] = []

        for tracklet_index in range(len(self._missing_tracklets)):
            self._missing_tracklets_counter[tracklet_index] += 1
            unmatched_cnt = self._missing_tracklets_counter[tracklet_index]

            if unmatched_cnt >= self._object_persist_frames:
                continue  # tracklet will be removed

            keep_indices.append(tracklet_index)

        # Remove outdated tracklets
        self._missing_tracklets = [self._missing_tracklets[t_i] for t_i in keep_indices]
        self._missing_tracklets_features = [self._missing_tracklets_features[t_i] for t_i in keep_indices]
        self._missing_tracklets_forecasts = [self._missing_tracklets_forecasts[t_i] for t_i in keep_indices]
        self._missing_tracklets_counter = [self._missing_tracklets_counter[t_i] for t_i in keep_indices]

    def _create_new_tracklets(self, detections: List[PredBBox]) -> None:
        """
        Initializes new tracklets for unmatched objects.

        Args:
            detections: Unmatched detections
        """
        for d in detections:
            d.label = self._create_next_object_index()
            self._active_tracklets.append(d)
            self._active_tracklets_forecasts.append([d])

            bbox_array = d.as_numpy_xyhw()
            initial_features = torch.from_numpy(bbox_array).view(1, 1, 4)
            self._active_tracklets_features.append(initial_features)

            logger.debug(f'Initialized new tracklet: {d.compact_repr}')

    @staticmethod
    def get_matches_tracklet_indices(matches: List[Tuple[int, int]]) -> List[int]:
        """
        Extracts tracklet indices from tracklet-detection matches

        Args:
            matches: (tracklet, detection)

        Returns:
            List of tracklet indices.
        """
        return [t_i for t_i, _ in matches]

    @staticmethod
    def get_matches_detection_indices(matches: List[Tuple[int, int]]) -> List[int]:
        """
        Extracts detection indices from tracklet-detection matches

        Args:
            matches: (tracklet, detection)

        Returns:
            List of detection indices.
        """
        return [d_i for _, d_i in matches]


    @staticmethod
    def _log_matches(
        tracklet_bboxes: List[PredBBox],
        detections: List[PredBBox],
        unmatched_tracklets_bboxes: List[PredBBox],
        unmatched_detections: List[PredBBox],
        tracklet_type: str
    ) -> None:
        """
        Add noisy logs (tracing) used for tracker debugging. Info:
        - Tracklets matches
        - Unmatched tracklets
        - Unmatched detections

        Args:
            tracklet_bboxes: tracklet bboxes
            detections: detection bboxes
            unmatched_tracklets_bboxes: unmatched tracklet bboxes
            unmatched_detections: unmatched detection bboxes
            tracklet_type: tracklet type (active/missing)
        """
        for tracklet_bbox, det_bbox in zip(tracklet_bboxes, detections):
            logger.debug(f'Matched ({tracklet_type}) tracklet {tracklet_bbox.compact_repr} with {det_bbox.compact_repr}')

        for bbox in unmatched_tracklets_bboxes:
            logger.debug(f'Unmatched ({tracklet_type}) tracklet: {bbox.compact_repr}')

        for bbox in unmatched_detections:
            logger.debug(f'Unmatched ({tracklet_type}) detection: {bbox.compact_repr}')

    def track(self, detections: List[PredBBox]) -> List[PredBBox]:
        """
        Updates tracklets with new detections

        Args:
            detections: List of new detected objects\\\

        Returns:
            List of tracklets for active and missing objects
        """
        logger.debug('---')  # Chunking debug log by iterations (adds space between chunks)
        logger.debug(f'Tracker iteration: {self._iter_index}')

        # Match detections with active tracklets
        estimated_active_tracklets_bboxes = self._predict_active()
        active_matches, unmatched_active_tracklets_indices, unmatched_active_detections_indices = \
            self._matcher(estimated_active_tracklets_bboxes, detections)

        # Extract active matches info (TODO: Maybe refactor? - duplicate code)
        matched_active_tracklet_indices = self.get_matches_tracklet_indices(active_matches)
        matched_active_detections = [detections[d_i] for d_i in self.get_matches_detection_indices(active_matches)]
        matched_active_tracklet_bboxes = [self._active_tracklets[t_i] for t_i in matched_active_tracklet_indices]
        unmatched_active_tracklets = [self._active_tracklets[t_i] for t_i in unmatched_active_tracklets_indices]
        unmatched_active_detections = [detections[d_i] for d_i in unmatched_active_detections_indices]
        self._log_matches(matched_active_tracklet_bboxes, matched_active_detections,
                          unmatched_active_tracklets, unmatched_active_detections, tracklet_type='active')

        # Update active trackelts with new matches
        self._update_active_tracklets(matched_active_tracklet_indices, matched_active_detections)

        # Match remaining detections with missing tracklets
        remaining_detections = [detections[ud_index] for ud_index in unmatched_active_detections_indices]
        estimated_missing_tracklets_bboxes = self._predict_missing()
        missing_matches, unmatched_missing_trackelts_indices, unmatched_missing_detections_indices = \
            self._matcher(estimated_missing_tracklets_bboxes, remaining_detections)
        # TODO: Update coords for missing matches (features should not be updated)

        # Extract missing matches info
        matched_missing_tracklets_indices = self.get_matches_tracklet_indices(missing_matches)
        matched_missing_detections = [remaining_detections[d_i] for d_i in self.get_matches_detection_indices(missing_matches)]
        matched_missing_tracklet_bboxes = [self._missing_tracklets[t_i] for t_i in matched_missing_tracklets_indices]
        unmatched_missing_tracklets = [self._missing_tracklets[t_i] for t_i in unmatched_missing_trackelts_indices]
        unmatched_missing_detections = [remaining_detections[d_i] for d_i in unmatched_missing_detections_indices]
        self._log_matches(matched_missing_tracklet_bboxes, matched_missing_detections,
                          unmatched_missing_tracklets, unmatched_missing_detections, tracklet_type='missing')

        # Move matched missing tracklets to active tracklets
        self._move_tracklets_to_active(matched_missing_tracklets_indices, matched_missing_detections)

        # Move unmatched active tracklets to missing tracklets
        self._move_tracklets_to_missing(unmatched_active_tracklets_indices)
        unmatched_detections = [remaining_detections[ud_index] for ud_index in unmatched_missing_detections_indices]

        # Update missing tracklets bboxes
        self._update_missing_tracklets()

        # Add unmatched detections to active tracklets
        # TODO: New objects should now appear immediately? FP
        # TODO: Update features for new objects
        self._create_new_tracklets(unmatched_detections)

        logger.debug(f'Stats: Number of active tracklets is {len(self._active_tracklets)}'
                     f' and number of missing tracklets is {len(self._missing_tracklets)}.')
        self._iter_index += 1
        return deepcopy(self._active_tracklets)
