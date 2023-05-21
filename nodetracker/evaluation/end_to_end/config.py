from dataclasses import dataclass, field
from nodetracker.config_parser import GlobalConfig
from typing import Optional


@dataclass
class FilterConfig:
    type: str
    params: dict


@dataclass
class JitterConfig:
    detection_noise_sigma: float = field(default=0.0)
    detection_skip_proba: float = field(default=0.0)


@dataclass
class Evaluation:
    n_steps: int = field(default=5)
    skip_occlusion: bool = field(default=False)
    skip_out_of_view: bool = field(default=False)
    occlusion_as_skip_detection: bool = field(default=True)


@dataclass
class Visualization:
    enable: bool = field(default=False)
    show_iou: bool = field(default=True)


@dataclass
class SelectionFilter:
    scene: Optional[str] = field(default=None)


@dataclass
class E2EConfig:
    filter: FilterConfig

    jitter: JitterConfig = field(default_factory=JitterConfig)
    eval: Evaluation = field(default_factory=Evaluation)
    visualization: Visualization = field(default_factory=Visualization)
    selection: SelectionFilter = field(default_factory=SelectionFilter)


@dataclass
class ExtendedE2EGlobalConfig(GlobalConfig):
    end_to_end: Optional[E2EConfig] = field(default=None)

    def __post_init__(self) -> None:
        """
        Validation
        """
        assert self.end_to_end is not None, 'E2E config needs to be defined!'
