"""
Implementation of tracklets-detections association matching algorithms.

Supports:
    - HungarianAlgorithmIOU
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Dict, Union

import numpy as np
import scipy

from nodetracker.library.cv.bbox import PredBBox

LabelType = Union[int, str]


class AssociationAlgorithm(ABC):
    """
    Defines interface to tracklets-detections association matching.
    """
    @abstractmethod
    def match(self, tracklets: List[PredBBox], detections: List[PredBBox]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Performs matching between tracklets and detections (interface).

        Args:
            tracklets: Tracked object from previous frames
            detections: Currently detected objects

        Returns:
            - List of matches (pairs)
            - list of unmatched tracklets
            - list of unmatched detections
        """
        pass

    def __call__(self, tracklets: List[PredBBox], detections: List[PredBBox]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        return self.match(tracklets, detections)


class HungarianAlgorithmIOU(AssociationAlgorithm):
    """
    Solves the linear sum assignment problem from given cost matrix based on IOU scores.
    """
    def __init__(
        self,
        match_threshold: float = 0.30,
        label_gating: Optional[Union[LabelType, List[Tuple[LabelType, LabelType]]]] = None,
        *args, **kwargs
    ):
        """
        Args:
            match_threshold: Min threshold do match tracklet with object.
                If threshold is not met then cost is equal to infinity.
        """
        super().__init__(*args, **kwargs)

        self._match_threshold = match_threshold
        self._label_gating = set([tuple(e) for e in label_gating]) if label_gating is not None else None
        self._INF = 999999  # Big number but less than np.inf

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

    def match(self, tracklets: List[PredBBox], detections: List[PredBBox]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        n_tracklets, n_detections = len(tracklets), len(detections)
        cost_matrix = np.zeros(shape=(n_tracklets, n_detections), dtype=np.float32)
        for t_i in range(n_tracklets):
            tracklet_bbox = tracklets[t_i]

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
                score = -iou_score if iou_score > self._match_threshold else np.inf
                cost_matrix[t_i][d_i] = score

        # Augment cost matrix with fictional values so that the linear_sum_assignment always has some solution
        augmentation = self._INF * np.ones(shape=(n_tracklets, n_tracklets), dtype=np.float32) - np.eye(n_tracklets, dtype=np.float32)
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
        'hungarian_iou': HungarianAlgorithmIOU
    }

    if name not in catalog:
        raise ValueError(f'Unknown algorithm "{name}". Available: {list(catalog.keys())}')

    return catalog[name](**params)
