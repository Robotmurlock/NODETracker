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
class TrackerVisualizeConfig:
    fps: int = 20
    new_object_length: int = 5
    option: str = 'active'

    def __post_init__(self) -> None:
        """
        Validation.
        """
        options = ['active', 'all', 'postprocess']
        assert self.option in options, f'Invalid option "{self.option}". Available: {options}.'


@dataclass
class TrackerPostprocessConfig:
    init_threshold: int = 2  # Activate `init_threshold` starting bboxes
    linear_interpolation_threshold: int = 3  # Maximum distance to perform linear interpolation
    linear_interpolation_min_tracklet_length: int = 0  # Minimum tracklet length required for interpolation
    min_tracklet_length: int = 20  # Remove all tracklets that are shorter than this



@dataclass
class TrackerConfig:
    object_detection: ObjectDetectionInferenceConfig
    lookup_path: str
    algorithm: TrackerAlgorithmConfig
    scene_pattern: str = '(.*?)'  # All
    output_path: str = 'tracker_inference'
    suffix: Optional[str] = None
    postprocess: TrackerPostprocessConfig = field(default_factory=TrackerPostprocessConfig)
    visualize: TrackerVisualizeConfig = field(default_factory=TrackerVisualizeConfig)


@dataclass
class TrackerGlobalConfig(GlobalConfig):
    tracker: Optional[TrackerConfig] = field(default=None)

    def __post_init__(self) -> None:
        """
        Validation
        """
        super().__post_init__()
        assert self.tracker is not None, 'E2E config needs to be defined!'
