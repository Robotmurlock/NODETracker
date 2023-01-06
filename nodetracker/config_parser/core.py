"""
Config structure. Config should be loaded as dictionary and parsed into GlobalConfig Python object. Benefits:
- Structure and type validation (using dacite library)
- Custom validations
- Python IDE autocomplete
"""
from dataclasses import dataclass
from typing import Optional

import dacite


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
    path: str
    metric_monitor: str
    resume_from: Optional[str]


@dataclass
class TrainConfig:
    experiment: str
    batch_size: int
    max_epochs: int
    learning_rate: float

    logging_cfg: TrainLoggingConfig
    checkpoint_cfg: TrainCheckpointConfig

@dataclass
class EvalConfig:
    experiment: str
    batch_size: int
    num_workers: int
    inference_name: str
    split: str
    checkpoint: str


@dataclass
class GlobalConfig:
    """
    Scripts GlobalConfig
    """
    resources: ResourcesConfig
    dataset: DatasetConfig
    train: TrainConfig
    eval: EvalConfig
    model: ModelConfig

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
