"""
SortTracker debug test (few iteration)
"""
import logging

from nodetracker.library.cv.bbox import PredBBox, Point
from nodetracker.node.kalman_filter import TorchConstantVelocityODKalmanFilter
from nodetracker.sort import SortTracker
from nodetracker.sort.matching import HungarianAlgorithmIOU
from nodetracker.utils.logging import configure_logging


def run_test() -> None:
    configure_logging(logging.DEBUG)

    forecaster = TorchConstantVelocityODKalmanFilter(
        time_step_multiplier=3e-2,
        process_noise_multiplier=1,
        measurement_noise_multiplier=1
    )
    matcher = HungarianAlgorithmIOU(match_threshold=0.3)
    tracker = SortTracker(forecaster, matcher)

    fst_iter_detections = [
        PredBBox(
            conf=0.9,
            upper_left=Point(x=0.1, y=0.1),
            bottom_right=Point(x=0.2, y=0.2)
        ),
        PredBBox(
            conf=0.9,
            upper_left=Point(x=0.5, y=0.5),
            bottom_right=Point(x=0.7, y=0.7)
        )
    ]

    snd_iter_detections = [
        PredBBox(
            conf=0.9,
            upper_left=Point(x=0.11, y=0.11),
            bottom_right=Point(x=0.21, y=0.21)
        ),
        PredBBox(
            conf=0.9,
            upper_left=Point(x=0.51, y=0.51),
            bottom_right=Point(x=0.71, y=0.71)
        ),
        PredBBox(
            conf=0.9,
            upper_left=Point(x=0.6, y=0.1),
            bottom_right=Point(x=0.7, y=0.2)
        )
    ]

    third_iter_detections = [
        PredBBox(
            conf=0.9,
            upper_left=Point(x=0.12, y=0.12),
            bottom_right=Point(x=0.22, y=0.22)
        ),
        PredBBox(
            conf=0.9,
            upper_left=Point(x=0.52, y=0.52),
            bottom_right=Point(x=0.72, y=0.72)
        )
    ]

    fourth_iter_detections = [
        PredBBox(
            conf=0.9,
            upper_left=Point(x=0.13, y=0.13),
            bottom_right=Point(x=0.23, y=0.23)
        )
    ]

    fifth_iter_detections = [
        PredBBox(
            conf=0.9,
            upper_left=Point(x=0.15, y=0.15),
            bottom_right=Point(x=0.25, y=0.25)
        ),
        PredBBox(
            conf=0.9,
            upper_left=Point(x=0.55, y=0.55),
            bottom_right=Point(x=0.75, y=0.75)
        )
    ]

    print('First iteration:', tracker.track(fst_iter_detections))  # 2 new objects
    print('Second iteration:', tracker.track(snd_iter_detections))  # 2 same objects, 1 FP
    print('Third iteration:', tracker.track(third_iter_detections))  # 2 same objects
    print('Fourth iteration:', tracker.track(fourth_iter_detections))  # 2nd object missing
    print('Fifth iteration:', tracker.track(fifth_iter_detections))  # 2nd object back


if __name__ == '__main__':
    run_test()
