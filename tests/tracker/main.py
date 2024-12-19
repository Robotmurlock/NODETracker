import unittest

from nodetracker.library.cv.bbox import PredBBox, Point
from nodetracker.tracker.trackers.byte import ByteTracker
from nodetracker.tracker.trackers.sort import SortTracker
from nodetracker.tracker import Tracklet, TrackletState
from nodetracker.tracker.matching import DCMIoU


class TestByte(unittest.TestCase):
    def test_bytetrack(self):
        # Arrange
        tracker = ByteTracker(
            filter_name='akf',
            filter_params={},
            remember_threshold=10,
            initialization_threshold=2,
            new_tracklet_detection_threshold=0.7
        )
        tracklets = []

        detections = [
            [
                PredBBox(
                    upper_left=Point(0.1, 0.1),
                    bottom_right=Point(0.2, 0.2),
                    label=0,
                    conf=0.9
                ),
                PredBBox(
                    upper_left=Point(0.3, 0.3),
                    bottom_right=Point(0.5, 0.5),
                    label=0,
                    conf=0.9
                ),
                PredBBox(
                    upper_left=Point(0.8, 0.8),
                    bottom_right=Point(0.9, 0.9),
                    label=0,
                    conf=0.9
                )
            ],
            [
                PredBBox(
                    upper_left=Point(0.1, 0.1),
                    bottom_right=Point(0.2, 0.2),
                    label=0,
                    conf=0.9
                ),
                PredBBox(
                    upper_left=Point(0.9, 0.1),
                    bottom_right=Point(0.95, 0.15),
                    label=0,
                    conf=0.2
                ),
                PredBBox(
                    upper_left=Point(0.31, 0.31),
                    bottom_right=Point(0.49, 0.49),
                    label=0,
                    conf=0.9
                ),
                PredBBox(
                    upper_left=Point(0.0, 0.0),
                    bottom_right=Point(0.01, 0.01),
                    label=0,
                    conf=0.7
                ),
                PredBBox(
                    upper_left=Point(0.79, 0.79),
                    bottom_right=Point(0.91, 0.91),
                    label=0,
                    conf=0.1
                )
            ],
            [
                PredBBox(
                    upper_left=Point(0.1, 0.1),
                    bottom_right=Point(0.2, 0.2),
                    label=0,
                    conf=0.9
                ),
                PredBBox(
                    upper_left=Point(0.31, 0.31),
                    bottom_right=Point(0.49, 0.49),
                    label=0,
                    conf=0.3
                ),
            ],
            [
                PredBBox(
                    upper_left=Point(0.1, 0.1),
                    bottom_right=Point(0.2, 0.2),
                    label=0,
                    conf=0.9
                ),
                PredBBox(
                    upper_left=Point(0.31, 0.31),
                    bottom_right=Point(0.49, 0.49),
                    label=0,
                    conf=0.9
                ),
                PredBBox(
                    upper_left=Point(0.79, 0.79),
                    bottom_right=Point(0.91, 0.91),
                    label=0,
                    conf=0.1
                )
            ]
        ]

        expected_active_count = [0, 3, 2, 2]
        active_count = []
        expected_all_count = [3, 4, 4, 4]
        all_count = []

        # Act
        for index, ds in enumerate(detections):
            active_tracklets, tracklets = tracker.track(tracklets, ds, frame_index=index)
            active_count.append(len(active_tracklets))
            all_count.append(len(tracklets))

        # Assert
        self.assertEquals(expected_active_count, active_count)
        self.assertEquals(expected_all_count, all_count)

    def test_sort_with_dcm(self):
        # Arrange
        tracker = SortTracker(
            filter_name='akf',
            filter_params={},
            remember_threshold=10,
            initialization_threshold=2,
            new_tracklet_detection_threshold=0.7,
            matcher_algorithm='dcm',
            matcher_params={
                'levels': 3
            }
        )
        tracklets = []

        detections = [
            [
                PredBBox(
                    upper_left=Point(0.1, 0.1),
                    bottom_right=Point(0.2, 0.2),
                    label=0,
                    conf=0.9
                ),
                PredBBox(
                    upper_left=Point(0.31, 0.31),
                    bottom_right=Point(0.5, 0.5),
                    label=0,
                    conf=0.9
                ),
                PredBBox(
                    upper_left=Point(0.8, 0.8),
                    bottom_right=Point(0.9, 0.9),
                    label=0,
                    conf=0.9
                )
            ],
            [
                PredBBox(
                    upper_left=Point(0.1, 0.1),
                    bottom_right=Point(0.2, 0.2),
                    label=0,
                    conf=0.9
                ),
                PredBBox(
                    upper_left=Point(0.31, 0.31),
                    bottom_right=Point(0.49, 0.49),
                    label=0,
                    conf=0.9
                )
            ]
        ]

        expected_active_count = [0, 2]
        active_count = []
        expected_all_count = [3, 2]
        all_count = []

        # Act
        for index, ds in enumerate(detections):
            active_tracklets, tracklets = tracker.track(tracklets, ds, frame_index=index)
            active_count.append(len(active_tracklets))
            all_count.append(len(tracklets))

        # Assert
        self.assertEquals(expected_active_count, active_count)
        self.assertEquals(expected_all_count, all_count)

    def test_dcm(self):
        # Arrange
        STEP_X, STEP_Y = 0.02, 0.04
        SIZE_X, SIZE_Y = 0.01, 0.02
        bboxes = [
            PredBBox(
                upper_left=Point((i + 1) * STEP_X, (i + 1) * STEP_Y),
                bottom_right=Point((i + 1) * STEP_X + SIZE_X, (i + 1) * STEP_Y + SIZE_Y),
                label=0,
                conf=0.9
            ) for i in range(20)
        ]

        tracklet_bboxes = [bboxes[i] for i in range(15)]
        detection_bboxes = [bboxes[2 * i] for i in range(10)]

        matcher = DCMIoU(levels=3)

        expected_matches = {(2 * i, i) for i in range(8)}
        expected_unmatched_tracklet_indices = {2 * i + 1 for i in range(7)}
        expected_unmatched_detection_indices = {8, 9}

        # Act
        matches, unmatched_tracklet_indices, unmatched_detection_indices = matcher(tracklet_bboxes, detection_bboxes)

        # Assert
        self.assertEquals(expected_matches, set(matches))
        self.assertEquals(expected_unmatched_tracklet_indices, set(unmatched_tracklet_indices))
        self.assertEquals(expected_unmatched_detection_indices, set(unmatched_detection_indices))
