"""
Implementation of tracklets-detections association matching algorithms.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import scipy

from nodetracker.library.cv.bbox import PredBBox, Point


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
    def __init__(self, match_threshold: float):
        """
        Args:
            match_threshold: Min threshold do match tracklet with object.
                If threshold is not met then cost is equal to infinity.
        """
        self._match_threshold = match_threshold
        self._INF = 999

    def match(self, tracklets: List[PredBBox], detections: List[PredBBox]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        n_tracklets, n_detections = len(tracklets), len(detections)
        cost_matrix = np.zeros(shape=(n_tracklets, n_detections), dtype=np.float32)
        for t_i in range(n_tracklets):
            for d_i in range(n_detections):
                tracklet_bbox = tracklets[t_i]
                det_bbox = detections[d_i]
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


def run_test():
    matcher = HungarianAlgorithmIOU(0.2)
    tracklets = [
        PredBBox(  # matches with D0
            label=1,
            upper_left=Point(x=0.1, y=0.1),
            bottom_right=Point(x=0.3, y=0.3)
        ),
        PredBBox(  # matches with D2
            label=2,
            upper_left=Point(x=0.7, y=0.9),
            bottom_right=Point(x=0.8, y=0.95)
        ),
        PredBBox(  # does not match
            label=0,
            upper_left=Point(x=0.0, y=0.0),
            bottom_right=Point(x=0.01, y=0.01)
        )
    ]
    detections = [
        PredBBox(  # matches with T0
            label=0,
            upper_left=Point(x=0.11, y=0.09),
            bottom_right=Point(x=0.29, y=0.32)
        ),
        PredBBox(  # does not match
            label=0,
            upper_left=Point(x=0.1, y=0.9),
            bottom_right=Point(x=0.2, y=0.8)
        ),
        PredBBox(  # matches T1
            label=0,
            upper_left=Point(x=0.65, y=0.9),
            bottom_right=Point(x=0.8, y=0.95)
        ),
        PredBBox(  # does not match
            label=0,
            upper_left=Point(x=0.99, y=0.99),
            bottom_right=Point(x=1.00, y=1.00)
        )
    ]

    output = matcher.match(tracklets, detections)
    expected_output = ([(0, 0), (1, 2)], [2], [1, 3])
    assert output == expected_output, f'Expected {expected_output} but found {output}!'


if __name__ == '__main__':
    run_test()
