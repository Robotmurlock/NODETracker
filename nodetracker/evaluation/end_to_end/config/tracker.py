"""
Extension of core config that supports additional parameters for E2E evaluation.
"""
from dataclasses import dataclass, field
from typing import Optional

from nodetracker.config_parser import GlobalConfig
from nodetracker.evaluation.end_to_end.config.common import ObjectDetectionInferenceConfig


@dataclass
class TrackerAlgorithmConfig:
    name: str
    params: dict


@dataclass
class TrackerConfig:
    object_detection: ObjectDetectionInferenceConfig
    lookup_path: str
    algorithm: TrackerAlgorithmConfig
    output_path: str

@dataclass
class TrackerGlobalConfig(GlobalConfig):
    tracker: Optional[TrackerConfig] = field(default=None)

    def __post_init__(self) -> None:
        """
        Validation
        """
        super().__post_init__()
        assert self.tracker is not None, 'E2E config needs to be defined!'
