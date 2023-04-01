"""
Config structure. Config should be loaded as dictionary and parsed into GlobalConfig Python object. Benefits:
- Structure and type validation (using dacite library)
- Custom validations
- Python IDE autocomplete
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Union, Tuple
from nodetracker.datasets.augmentations import create_identity_augmentation_config, TrajectoryAugmentation
from nodetracker.utils.serialization import serialize_json
from hydra.utils import instantiate
from omegaconf import OmegaConf

import dacite
import yaml

from nodetracker.common import project


@dataclass
class DatasetConfig:
    """
    Dataset config.
    - name: Dataset name
    - train_path: Path to training dataset subset
    - val_path: Path to validation dataset subset
    - test_path: Path to test dataset subset
    - history_len: Number of observable input values
    - future_len: Number of unobservable output values (values that are being predicted)
    """
    name: str
    train_path: str
    val_path: str
    test_path: str
    history_len: int
    future_len: int

    def get_split_path(self, split: str) -> str:
        """
        Gets split path.

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
    """
    Features transform config. Used to transform input features. Optionally performs inverse operation to output values.
    - name: Name of transform class (type).
    - params: Transform parameters
    """
    name: str
    params: dict


@dataclass
class AugmentationsConfig:
    """
    Augmentations applied before or after transform:
    - before_transform: Composition of augmentations applied before transform function.
    - after_transform: Composition of augmentations applied after transform function.
    """
    before_transform_config: Optional[dict]
    after_transform_config: Optional[dict]
    after_batch_collate_config: Optional[dict]

    def __post_init__(self):
        """
        Validation: Check augmentation object instantiation
        """
        self.before_transform_config, self.after_transform_config, self.after_batch_collate_config = \
            [create_identity_augmentation_config() if cfg is None else cfg
             for cfg in [self.before_transform_config, self.after_transform_config, self.after_batch_collate_config]]

        self.before_transform = instantiate(OmegaConf.create(self.before_transform_config))
        self.after_transform = instantiate(OmegaConf.create(self.after_transform_config))
        self.after_batch_collate = instantiate(OmegaConf.create(self.after_batch_collate_config))

    @classmethod
    def default(cls) -> 'AugmentationsConfig':
        """
        Default augmentations (none) in case it is not defined.

        Returns:
            Default augmentations
        """
        return cls(
            before_transform_config=create_identity_augmentation_config(),
            after_transform_config=create_identity_augmentation_config(),
            after_batch_collate_config=create_identity_augmentation_config()
        )

@dataclass
class ModelConfig:
    """
    Model config:
    - type: Model (architecture) type
    - params: Model creation parameters
    """
    type: str
    params: dict


@dataclass
class ResourcesConfig:
    """
    Resources config (cpu/gpu, number of cpu cores, ...)
    - gpus: Number of gpus
    - accelerator: gpu/cpu
    - num_workers: cpu workers
    """
    gpus: int
    accelerator: str
    num_workers: int


@dataclass
class TrainLoggingConfig:
    """
    Configs for script logging during model training (not important for inference).
    - path: TB logs path (deprecated)
    - log_every_n_steps: TB log frequency
    """
    path: str  # Deprecated (predefined by conventions)
    log_every_n_steps: int


@dataclass
class TrainCheckpointConfig:
    """
    Model checkpoint saving config.
    - metric_monitor: Chooses the best checkpoint based on metric name
    - resume_from: Start from chosen checkpoint (fine-tuning)
    """
    metric_monitor: str
    resume_from: Optional[str]


@dataclass
class TrainConfig:
    """
    Train configuration.
    - experiment: Name of the training experiment
    - description: Experiment description
    - batch_size: Training batch size
    - max_epochs: Number of epochs to train model

    - logging_cfg: TrainLoggingConfig
    - checkpoint_cfg: TrainCheckpointConfig

    - train_params: Training architecture specific parameters
    """
    experiment: str
    description: str
    batch_size: int
    max_epochs: int

    logging_cfg: TrainLoggingConfig
    checkpoint_cfg: TrainCheckpointConfig

    train_params: Optional[dict] = field(default_factory=dict)  # default: empty dict


@dataclass
class EvalConfig:
    """
    Inference + Evaluation config
    - experiment: experiment name (should match train)
    - batch_size: inference batch size
    - inference_name: inference run name
    - split: what split to use for evaluation (train/val/test)
    - checkpoint: what checkpoint to use for evaluation

    - autoregressive: use autoregressive decorator
    - autoregressive_keep_history: requires `autoregressive` - keeps all history when predicting (not droppping last)
    """
    experiment: str
    batch_size: int
    inference_name: str
    split: str
    checkpoint: Optional[str]

    # Autoregressive configs
    autoregressive: bool = field(default=False)
    autoregressive_keep_history: bool = field(default=False)


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
    """
    Path configs
    - master: location where all final and intermediate results are stored
    - assets: location where datasets can be found
    """
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

    augmentations: AugmentationsConfig = field(default_factory=AugmentationsConfig.default)
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
            data = asdict(self)  # Not fully "serialized"
            data = serialize_json(data)
            yaml.safe_dump(data, f)
