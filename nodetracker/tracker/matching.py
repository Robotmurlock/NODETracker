"""
Implementation of tracklets-detections association matching algorithms.

Supports:
    - HungarianAlgorithmIOU
    - Byte
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Dict, Union

import numpy as np
import scipy

from nodetracker.library.cv.bbox import PredBBox
from nodetracker.tracker.tracklet import Tracklet, TrackletState

LabelType = Union[int, str]
INF = 999_999


class AssociationAlgorithm(ABC):
    """
    Defines interface to tracklets-detections association matching.
    """
    @abstractmethod
    def match(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        tracklets: Optional[List[Tracklet]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Performs matching between tracklets and detections (interface).

        Args:
            tracklet_estimations: Tracked object from previous frames
            detections: Currently detected objects
            tracklets: Full tracklet info (optional)

        Returns:
            - List of matches (pairs)
            - list of unmatched tracklets
            - list of unmatched detections
        """
        pass

    def __call__(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        tracklets: Optional[List[Tracklet]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        return self.match(tracklet_estimations, detections, tracklets=tracklets)


def hungarian(cost_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Performs Hungarian algorithm on arbitrary `cost_matrix`.

    Args:
        cost_matrix: Any cost matrix

    Returns:
        - List of matched (tracklet, detection) index pairs
        - List of unmatched tracklets indices
        - List of unmatched detection indices
    """
    n_tracklets, n_detections = cost_matrix.shape

    # Augment cost matrix with fictional values so that the linear_sum_assignment always has some solution
    augmentation = INF * np.ones(shape=(n_tracklets, n_tracklets), dtype=np.float32) - np.eye(n_tracklets, dtype=np.float32)
    augmented_cost_matrix = np.hstack([cost_matrix, augmentation])

    row_indices, col_indices = scipy.optimize.linear_sum_assignment(augmented_cost_matrix)

    # All row indices that have values above are matched with augmented part of the matrix hence they are not matched
    matches = []
    matched_tracklets = set()
    matched_detections = set()
    for r_i, c_i in zip(row_indices, col_indices):
        if c_i >= n_detections:
            continue  # augmented match -> no match
        matches.append((r_i, c_i))
        matched_tracklets.add(r_i)
        matched_detections.add(c_i)

    unmatched_tracklets = list(set(range(n_tracklets)) - matched_tracklets)
    unmatched_detections = list(set(range(n_detections)) - matched_detections)

    return matches, unmatched_tracklets, unmatched_detections


class HungarianAlgorithmIOU(AssociationAlgorithm):
    """
    Solves the linear sum assignment problem from given cost matrix based on IOU scores.
    """
    def __init__(
        self,
        match_threshold: float = 0.30,
        fuse_score: bool = False,
        label_gating: Optional[Union[LabelType, List[Tuple[LabelType, LabelType]]]] = None,
        *args, **kwargs
    ):
        """
        Args:
            match_threshold: Min threshold do match tracklet with object.
                If threshold is not met then cost is equal to infinity.
            fuse_score: Fuse score with iou
            label_gating: Define which object labels can be matched
                - If not defined, matching is label agnostic
        """
        super().__init__(*args, **kwargs)

        self._match_threshold = match_threshold
        self._fuse_score = fuse_score

        self._label_gating = set([tuple(e) for e in label_gating]) if label_gating is not None else None

    def _can_match(self, tracklet_label: LabelType, det_label: LabelType) -> bool:
        """
        Checks if matching between tracklet and detection is possible.

        Args:
            tracklet_label: Tracklet label
            det_label: Detection label

        Returns:
            True if matching is possible else False
        """
        if tracklet_label == det_label:
            # Objects with same label can always match
            return True

        if self._label_gating is None:
            # If label gating is not set then any objects with same label can't match
            return False

        return (tracklet_label, det_label) in self._label_gating \
            or (det_label, tracklet_label) in self._label_gating

    def _form_iou_cost_matrix(self, tracklet_estimations: List[PredBBox], detections: List[PredBBox]) -> np.ndarray:
        """
        Creates negative IOU cost matrix as an input into Hungarian algorithm.

        Args:
            tracklet_estimations: List of tracklet estimated bboxes
            detections: Detection (observation) bboxes

        Returns:
            Negative IOU cost matrix
        """
        n_tracklets, n_detections = len(tracklet_estimations), len(detections)
        cost_matrix = np.zeros(shape=(n_tracklets, n_detections), dtype=np.float32)
        for t_i in range(n_tracklets):
            tracklet_bbox = tracklet_estimations[t_i]

            for d_i in range(n_detections):
                det_bbox = detections[d_i]

                # Check if matching is possible
                if not self._can_match(tracklet_bbox.label, det_bbox.label):
                    cost_matrix[t_i][d_i] = np.inf
                    continue

                # Calculate IOU score
                iou_score = tracklet_bbox.iou(det_bbox)
                # Higher the IOU score the better is the match (using negative values because of min optim function)
                # If score has very high value then
                score = iou_score * det_bbox.conf if self._fuse_score else iou_score
                score = - score if score > self._match_threshold else np.inf
                cost_matrix[t_i][d_i] = score

        return cost_matrix

    def match(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        tracklets: Optional[List[Tracklet]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        cost_matrix = self._form_iou_cost_matrix(tracklet_estimations, detections)
        return hungarian(cost_matrix)


class Byte(AssociationAlgorithm, ABC):
    """
    ByteTrack matching algorithm.

    Reference: https://arxiv.org/abs/2110.06864
    """
    def __init__(
        self,
        detection_threshold: float = 0.60,
        *args, **kwargs
    ):
        """
        Args:
            detection_threshold: Detection threshold between "high priority" detections and "low priority" detections.
                - Matching is first performed between all tracklets and detections with high detection scores.
                - Mathing is then performed between unmatched tracklets and detections with low detection scores
        """
        super().__init__(*args, **kwargs)

        self._detection_threshold = detection_threshold

    @abstractmethod
    def _match(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        tracklets: Optional[List[Tracklet]] = None,
        high: bool = True
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Custom matcher function that can be integrated in the Byte algorithm.

        Args:
            tracklet_estimations: Tracked object from previous frames
            detections: Currently detected objects
            tracklets: Full tracklet info (optional)
            high: Perform high or low matching

        Returns:
            - List of matches (pairs)
            - list of unmatched tracklets
            - list of unmatched detections
        """
        pass

    def match(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        tracklets: Optional[List[Tracklet]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        high_detections = [d for d in detections if d.conf >= self._detection_threshold]
        high_det_indices = [i for i, d in enumerate(detections) if d.conf >= self._detection_threshold]
        low_detections = [d for d in detections if d.conf < self._detection_threshold]
        low_det_indices = [i for i, d in enumerate(detections) if d.conf < self._detection_threshold]

        # Perform high detection matching
        high_matches, remaining_tracklet_indices, high_unmatched_detections = \
            self._match(tracklet_estimations, high_detections, tracklets, high=True)
        remaining_tracklet_estimations = [tracklet_estimations[t_i] for t_i in remaining_tracklet_indices
                                          if tracklets[t_i].state == TrackletState.ACTIVE]
        remaining_tracklets = [tracklets[t_i] for t_i in remaining_tracklet_indices if tracklets[t_i].state == TrackletState.ACTIVE]

        # Perform low detection matching
        low_matches, low_unmatched_tracklets, _ = \
            self._match(remaining_tracklet_estimations, low_detections, remaining_tracklets, high=False)

        # Map relative indices to the original ones
        high_matches = [(t_i, high_det_indices[d_i]) for t_i, d_i in high_matches]
        low_matches = [(remaining_tracklet_indices[t_i], low_det_indices[d_i]) for t_i, d_i in low_matches]
        matches = high_matches + low_matches

        unmatched_tracklets = [remaining_tracklet_indices[t_i] for t_i in low_unmatched_tracklets]

        high_unmatched_detections = [high_det_indices[d_i] for d_i in high_unmatched_detections]
        unmatched_detections = high_unmatched_detections  # Does not include low unmatched detections

        return matches, unmatched_tracklets, unmatched_detections


class ByteIOU(Byte):
    """
    Standard ByteTrack matching algorithm.
    """
    def __init__(
        self,
        detection_threshold: float = 0.60,
        match_threshold: float = 0.30,  # high_match_threshold
        low_match_threshold: float = 0.50,
        label_gating: Optional[Union[LabelType, List[Tuple[LabelType, LabelType]]]] = None,
        *args, **kwargs
    ):
        """
        Args:
            detection_threshold: Detection threshold between "high priority" detections and "low priority" detections.
                - Matching is first performed between all tracklets and detections with high detection scores.
                - Mathing is then performed between unmatched tracklets and detections with low detection scores
            match_threshold: Min threshold do match tracklet with object.
                If threshold is not met then cost is equal to infinity.
            label_gating: Define which object labels can be matched
                - If not defined, matching is label agnostic
        """
        super().__init__(detection_threshold=detection_threshold, *args, **kwargs)

        self._high_hungarian = HungarianAlgorithmIOU(
            match_threshold=match_threshold,
            label_gating=label_gating
        )

        self._low_hungarian = HungarianAlgorithmIOU(
            match_threshold=low_match_threshold,
            label_gating=label_gating
        )

    def _match(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        tracklets: Optional[List[Tracklet]] = None,
        high: bool = True
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if high:
            return self._high_hungarian(tracklet_estimations, detections, tracklets)
        return self._low_hungarian(tracklet_estimations, detections, tracklets)


def distance(name: str, x: np.ndarray, y: np.ndarray) -> float:
    if name == 'l1':
        return np.abs(x - y).sum()
    if name == 'l2':
        return np.sqrt(np.square(x - y).sum())

    raise AssertionError('Invalid Program State!')



class MotionAssoc(HungarianAlgorithmIOU):
    DISTANCE_OPTIONS = ['l1', 'l2']

    """
    Combines Hungarian IOU and motion estimation for tracklet and detection matching.
    """
    def __init__(
        self,
        match_threshold: float = 0.30,
        motion_lambda: float = 5,
        only_matched: bool = False,
        distance_name: str = 'l1',
        label_gating: Optional[Union[LabelType, List[Tuple[LabelType, LabelType]]]] = None,
        *args, **kwargs
    ):
        """
        Args:
            match_threshold: IOU match gating
            motion_lambda: Motion difference multiplier
            only_matched: Only use motion cost matrix for tracklets that are matched in last frame
            label_gating: Gating between different types of objects
        """
        super().__init__(
            match_threshold=match_threshold,
            label_gating=label_gating,
            *args, **kwargs
        )
        self._motion_lambda = motion_lambda
        self._only_matched = only_matched

        assert distance_name in self.DISTANCE_OPTIONS, f'Invalid distance option "{distance_name}". Available: {self.DISTANCE_OPTIONS}.'
        self._distance_name = distance_name

    def _form_motion_distance_cost_matrix(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        tracklets: Optional[List[Tracklet]] = None
    ) -> np.ndarray:
        """
        Forms motion matrix where motion is approximated as the difference
        between bbox and estimated tracklet motion.

        Args:
            tracklet_estimations: Tracklet bbox estimations
            detections: Detection bboxes
            tracklets: Tracklets full info (history, etc.)

        Returns:
            Motion distance cost matrix
        """
        n_tracklets, n_detections = len(tracklet_estimations), len(detections)
        cost_matrix = np.zeros(shape=(n_tracklets, n_detections), dtype=np.float32)
        for t_i in range(n_tracklets):
            tracklet_estimated_bbox = tracklet_estimations[t_i]
            tracklet_info = tracklets[t_i]
            is_matched = tracklet_info.state == TrackletState.ACTIVE

            tracklet_last_bbox = tracklet_info.bbox.as_numpy_xyxy()
            tracklet_motion = tracklet_estimated_bbox.as_numpy_xyxy() - tracklet_last_bbox

            for d_i in range(n_detections):
                det_bbox = detections[d_i]
                det_motion = det_bbox.as_numpy_xyxy() - tracklet_last_bbox
                cost_matrix[t_i][d_i] = distance(self._distance_name, tracklet_motion, det_motion)

            if self._only_matched and not is_matched:
                # Assumption: Motion estimation is not that accurate in this case
                cost_matrix[t_i, :] = 0
                continue

        return cost_matrix

    def match(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        tracklets: Optional[List[Tracklet]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        neg_iou_cost_matrix = self._form_iou_cost_matrix(tracklet_estimations, detections)
        motion_diff_cost_matrix = self._form_motion_distance_cost_matrix(tracklet_estimations, detections, tracklets)
        cost_matrix = neg_iou_cost_matrix + self._motion_lambda * motion_diff_cost_matrix
        return hungarian(cost_matrix)


class HungarianDistanceAndMotion(AssociationAlgorithm):
    DISTANCE_OPTIONS = ['l1', 'l2']

    def __init__(
        self,
        motion_lambda: float = 1,
        offset_gating_lambda: float = 1,
        distance_name: str = 'l1',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._motion_lambda = motion_lambda
        self._offset_gating_lambda = offset_gating_lambda

        assert distance_name in self.DISTANCE_OPTIONS, f'Invalid distance option "{distance_name}". Available: {self.DISTANCE_OPTIONS}.'
        self._distance_name = distance_name

    def _form_cost_matrix(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        tracklets: Optional[List[Tracklet]] = None
    ) -> np.ndarray:
        n_tracklets, n_detections = len(tracklet_estimations), len(detections)
        cost_matrix = np.zeros(shape=(n_tracklets, n_detections), dtype=np.float32)
        for t_i in range(n_tracklets):
            tracklet_estimated_bbox = tracklet_estimations[t_i]
            tracklet_info = tracklets[t_i]

            tracklet_last_bbox = tracklet_info.bbox.as_numpy_xyxy()
            tracklet_estimated_bbox_position = tracklet_estimated_bbox.as_numpy_xyxy()
            tracklet_motion = tracklet_estimated_bbox_position - tracklet_last_bbox

            for d_i in range(n_detections):
                det_bbox = detections[d_i]
                det_position = det_bbox.as_numpy_xyxy()
                det_motion = det_position - tracklet_last_bbox

                # Position cost matrix
                cost_matrix[t_i][d_i] += distance(self._distance_name, tracklet_estimated_bbox_position, det_position)

                # Motion cost matrix
                cost_matrix[t_i][d_i] += distance(self._distance_name, tracklet_motion, det_motion) * self._motion_lambda

                # Gating
                offset = np.abs(tracklet_estimated_bbox_position - det_position)
                h = tracklet_estimated_bbox_position[2] - tracklet_estimated_bbox_position[0]
                w = tracklet_estimated_bbox_position[3] - tracklet_estimated_bbox_position[1]
                if any([
                    offset[0] > h * self._offset_gating_lambda,
                    offset[1] > w * self._offset_gating_lambda,
                    offset[2] > h * self._offset_gating_lambda,
                    offset[3] > w * self._offset_gating_lambda
                ]):
                    cost_matrix[t_i][d_i] = np.inf

        return cost_matrix

    def match(
        self,
        tracklet_estimations: List[PredBBox],
        detections: List[PredBBox],
        tracklets: Optional[List[Tracklet]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        cost_matrix = self._form_cost_matrix(tracklet_estimations, detections, tracklets)
        return hungarian(cost_matrix)


def association_algorithm_factory(name: str, params: Dict[str, Any]) -> AssociationAlgorithm:
    """
    Association algorithm factory. Not case-sensitive.

    Args:
        name: Algorithm name
        params: Parameters

    Returns:
        AssociationAlgorithm object
    """
    name = name.lower()

    catalog = {
        'hungarian_iou': HungarianAlgorithmIOU,
        'byte': ByteIOU,
        'hungarian_iou_motion': MotionAssoc,
        'motion_assoc': MotionAssoc,  # alias
        'hungarian_distance_motion': HungarianDistanceAndMotion
    }

    if name not in catalog:
        raise ValueError(f'Unknown algorithm "{name}". Available: {list(catalog.keys())}')

    return catalog[name](**params)
