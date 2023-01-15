"""
MP4Writer support
"""
from typing import Tuple

import cv2
import numpy as np


class MP4Writer:
    """
    RAII implementation of cv2.VideoWriter
    """
    def __init__(self, path: str, fps: int, resolution: Tuple[int, int], resize: bool = True):
        """
        Args:
            path: video path
            fps: video fps
            resolution: video resolution
            resize: resize input images
        """
        self._path = path
        self._fps = fps
        self._resolution = resolution
        self._resize = resize

        # State
        self._writer = None

    def open(self) -> None:
        """
        Open video for writing.
        """
        # noinspection PyUnresolvedReferences
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # noinspection PyUnresolvedReferences
        self._writer = cv2.VideoWriter(self._path, fourcc, self._fps, self._resolution)

    def close(self) -> None:
        """
        Close video for writing.
        """
        self._writer.release()
        self._writer = None

    def write(self, image: np.ndarray) -> None:
        """
        Write image to video mp4.

        Args:
            image: Image
        """
        assert self._writer is not None, 'Writer not open.'
        h, w, _ = image.shape

        if (w, h) != self._resolution:
            assert self._resize, f'Got image resolution {(w, h)} but expected {self._resolution}'
            # noinspection PyUnresolvedReferences
            image = cv2.resize(image, self._resolution)

        self._writer.write(image)

    def __enter__(self) -> 'MP4Writer':
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
