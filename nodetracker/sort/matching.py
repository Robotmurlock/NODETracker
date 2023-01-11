from abc import ABC, abstractmethod
import numpy as np


class AssociationAlgorithm(ABC):
    @abstractmethod
    def match(self, tracklets: np.ndarray, detections: np.ndarray) -> np.ndarray:
        pass