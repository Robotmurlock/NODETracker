"""
Config structure. Config should be loaded as dictionary and parsed into GlobalConfig Python object. Benefits:
- Structure and type validation (using dacite library)
- Custom validations
- Python IDE autocomplete
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Union, Tuple

import dacite
import yaml

from nodetracker.common import project


@dataclass
class DatasetConfig:
    name: str
    train_path: str
    val_path: str
    test_path: str
    history_len: int
    future_len: int

    def get_split_path(self, split: str) -> str:
        """
        Gets split path

        Args:
            split: Split name (train, val, test)

        Returns:
            Path to split
        """
        valid_split_values = ['train', 'val', 'test']
        if split not in valid_split_values:
            raise ValueError(f'Invalid split "{split}". Available: {valid_split_values}')

        if split == 'train':
            return self.train_path
        elif split == 'val':
            return self.val_path
        elif split == 'test':
            return self.test_path
        else:
            raise AssertionError('Invalid Program State!')


@dataclass
class TransformConfig:
    name: str
    params: dict


@dataclass
class ModelConfig:
    type: str
    params: dict


@dataclass
class ResourcesConfig:
    gpus: int
    accelerator: str
    num_workers: int


@dataclass
class TrainLoggingConfig:
    path: str
    log_every_n_steps: int


@dataclass
class TrainCheckpointConfig:
    metric_monitor: str
    resume_from: Optional[str]


@dataclass
class TrainConfig:
    experiment: str
    description: str
    batch_size: int
    max_epochs: int

    logging_cfg: TrainLoggingConfig
    checkpoint_cfg: TrainCheckpointConfig

    train_params: Optional[dict] = field(default_factory=dict)  # default: empty dict
    resume_from_checkpoint: Optional[str] = field(default=None)

@dataclass
class EvalConfig:
    experiment: str
    batch_size: int
    num_workers: int
    inference_name: str
    split: str
    checkpoint: Optional[str]


@dataclass
class VisualizeConfig:
    """
    VisualizeConfig is extension of EvalConfig. Uses:
    - experiment
    - inference_name
    - split

    Also uses model_type from model configs.
    """
    resolution: Union[List[int], Tuple[int]]
    fps: int

    def __post_init__(self):
        """
        postprocess + validation
        """
        self.resolution = tuple(self.resolution)
        if len(self.resolution) != 2:
            raise ValueError(f'Invalid resolution: {self.resolution}')


@dataclass
class PathConfig:
    master: str
    assets: str

    @classmethod
    def default(cls) -> 'PathConfig':
        """
        Default path configuration is used if it is not defined in configs.

        Returns: Path configuration.
        """
        return cls(
            master=project.OUTPUTS_PATH,
            assets=project.ASSETS_PATH
        )


@dataclass
class GlobalConfig:
    """
    Scripts GlobalConfig
    """
    resources: ResourcesConfig
    dataset: DatasetConfig
    transform: TransformConfig
    train: TrainConfig
    eval: EvalConfig
    model: ModelConfig

    path: PathConfig = field(default_factory=PathConfig.default)
    visualize: Optional[VisualizeConfig] = None

    @classmethod
    def from_dict(cls, raw: dict) -> 'GlobalConfig':
        """
        Creates Config object form raw dictionary

        Args:
            raw: Raw dictionary config

        Returns:
            Parsed config
        """
        return dacite.from_dict(cls, raw)

    @property
    def prettyprint(self) -> str:
        """
        Pretty print.

        Returns:
            Yaml formatted config
        """
        return yaml.dump(asdict(self))

    def save(self, path: str) -> None:
        """
        Saves config to given path.

        Args:
            path: Path where to save config
        """
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(asdict(self), f)
