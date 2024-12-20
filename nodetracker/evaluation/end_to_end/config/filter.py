"""
Extension of core config that supports additional parameters for E2E evaluation.
"""
from dataclasses import dataclass, field
from typing import Optional

from nodetracker.config_parser import GlobalConfig
from nodetracker.evaluation.end_to_end.config.common import FilterConfig, ObjectDetectionInferenceConfig


@dataclass
class JitterConfig:
    detection_noise_sigma: float = field(default=0.0)
    detection_skip_proba: float = field(default=0.0)


@dataclass
class EvaluationConfig:
    n_steps: int = field(default=5)
    occlusion_as_skip_detection: bool = field(default=True)
    disable_eval_after_n_fns: Optional[int] = field(default=None)
    assoc_use_gt: bool = field(default=False)


@dataclass
class Visualization:
    enable: bool = field(default=False)
    prior: bool = field(default=True)
    posterior: bool = field(default=True)
    show_iou: bool = field(default=True)


@dataclass
class SelectionFilter:
    scene: Optional[str] = field(default=None)
    eval_split_only: bool = field(default=True)


@dataclass
class ESParams:
    max_skip_threshold: int = field(default=0)
    min_iou_match: float = field(default=0.0)


@dataclass
class E2EConfig:
    filter: FilterConfig
    object_detection: ObjectDetectionInferenceConfig
    lookup_path: str

    save_inference: bool = field(default=True)  # Only save prior and posterior
    jitter: JitterConfig = field(default_factory=JitterConfig)
    eval: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: Visualization = field(default_factory=Visualization)
    selection: SelectionFilter = field(default_factory=SelectionFilter)
    es: ESParams = field(default_factory=ESParams)


@dataclass
class FilterGlobalConfig(GlobalConfig):
    end_to_end: Optional[E2EConfig] = field(default=None)

    def __post_init__(self) -> None:
        """
        Validation
        """
        super().__post_init__()
        assert self.end_to_end is not None, 'E2E config needs to be defined!'
