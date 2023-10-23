import unittest

from nodetracker.library.cv.bbox import PredBBox, Point
from nodetracker.tracker.trackers.byte import ByteTracker


class TestByte(unittest.TestCase):
    def test_byte(self):
        # Arrange
        tracker = ByteTracker(
            filter_name='akf',
            filter_params={},
            remember_threshold=10
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

        expected_active_count = [0, 2, 2, 2]
        active_count = []
        expected_all_count = [3, 3, 2, 2]
        all_count = []

        # Act
        for index, ds in enumerate(detections):
            active_tracklets, tracklets = tracker.track(tracklets, ds, frame_index=index)
            active_count.append(len(active_tracklets))
            all_count.append(len(tracklets))

        # Assert
        self.assertEquals(expected_active_count, active_count)
        self.assertEquals(expected_all_count, all_count)
