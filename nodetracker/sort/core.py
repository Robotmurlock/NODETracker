from nodetracker.node import BBoxTrajectoryForecaster
from nodetracker.sort.matching import AssociationAlgorithm
import torch

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
        object_persist_frames: int
    ):
        """

        Args:
            forecaster: Forecasts bbox coords of tracked object in next frame
            object_persist_frames: Number of frames to track lost objects before removing them
        """
        self._forecaster = forecaster
        self._matcher = matcher

        self._active_objects = torch.zeros(0, 4, dtype=torch.float32)
        self._lost_objects = torch.zeros(0, 4, dtype=torch.float32)

        # Hyperparameters
        self._object_persist_frames = object_persist_frames

    def track(self):
        pass # TODO