"""
Tracklet Dataclass.
"""
from multiprocessing import Value, Lock
from typing import Tuple, ClassVar, Optional, List
from nodetracker.library.cv.bbox import PredBBox


_TRACKLET_DEFAULT_MAX_HISTORY = 8

TrackletHistoryType = List[Tuple[int, PredBBox]]


class Tracklet:
    """
    Tracking object state
    """
    Counter: ClassVar[Value] = Value('i', 0)
    CounterLock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        bbox: PredBBox,
        frame_index: int,
        max_history: int = _TRACKLET_DEFAULT_MAX_HISTORY,
        _id: Optional[int] = None
    ):
        """
        Initializes new tracklet.

        Args:
            bbox: Initial bbox
            frame_index: Frame index when tracklet was created
            max_history: Max tracklet history (old data is deleted)
            _id: Set tracklet if explicitly

        Returns:
            Initialized tracklet
        """
        if _id is None:
            with Tracklet.CounterLock:
                Tracklet.Counter.value += 1
                self._id = Tracklet.Counter.value
        else:
            self._id = _id

        # Update bbox
        bbox.id = self._id

        self._max_history = max_history

        # State
        self._history: TrackletHistoryType = [(frame_index, bbox)]
        self._total_matches = 1
        self._matched = True

    def __hash__(self) -> int:
        return self._id

    @property
    def matched(self) -> bool:
        """
        Checks if tracklet is matched in the update.

        Returns:
            True if tracklet is matched in last update else False.
        """
        return self._matched

    def number_of_unmatched_frames(self, current_frame_index: int) -> int:
        """
        Calculates number of frames that the tracklet was not matched with any of the detections
        counting from the last matched frames.

        Example: (match, missing, match, missing, missing) -> returns 2

        Args:
            current_frame_index: Current frame

        Returns:
            Number of unmatched frames
        """
        return current_frame_index - self.frame_index

    @property
    def id(self) -> int:
        """
        Gets tracklet id.

        Returns:
            Tracklet id
        """
        return self._id


    @property
    def latest(self) -> Tuple[int, PredBBox]:
        """
        The latest detection of one tracklet.

        Returns:
            id and PredBBox
        """
        return self._history[-1]

    @property
    def first(self) -> Tuple[int, PredBBox]:
        """
        First detection (last detection in history) of one tracklet.

        Returns:
            id and PredBBox
        """
        return self._history[0]

    @property
    def history(self) -> TrackletHistoryType:
        """
        Gets full tracklet history.

        Returns:
            Tracklet history
        """
        return self._history

    @property
    def bbox(self) -> PredBBox:
        """
        Gets current tracklet bbox.

        Returns:
            Current bbox
        """
        return self.latest[1]

    @property
    def frame_index(self) -> int:
        """
        Gets current (last) frame index.

        Returns:
            Frame index
        """
        return self.latest[0]

    @property
    def total_matches(self) -> int:
        """
        Total number of tracklet matches with detected object.

        Returns:
            Total number of tracklet matches
        """
        return self._total_matches

    def __len__(self) -> int:
        """
        Returns:
            tracklet length
        """
        return len(self._history)

    def update(self, bbox: PredBBox, frame_index: int, matched: bool = True) -> 'Tracklet':
        """
        Updates tracklet BBox (and history).
        Disclaimer: Updates bbox id!

        Args:
            bbox: New bbox
            frame_index: Current frame index
            matched: Is tracklet matched or not (updates counter if matched)
        """
        bbox.id = self._id  # Update bbox id

        # Remove `obsolete` history
        while len(self._history) > 1 and frame_index - self.first[0] > self._max_history:
            self._history.pop(0)

        self._history.append((frame_index, bbox))

        self._matched = matched
        if matched:
            self._total_matches += 1

        return self
