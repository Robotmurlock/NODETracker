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
    train_path: str
    val_path: str
    history_len: int
    future_len: int


@dataclass
class ModelTrainConfig:
    noise_std: float


@dataclass
class ModelConfig:
    observable_dim: int
    latent_dim: int
    hidden_dim: int
    train_cfg: ModelTrainConfig


@dataclass
class TrainerConfig:
    gpus: int
    accelerator: str


@dataclass
class TrainLoggingConfig:
    path: str
    name: str
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
    num_workers: int
    max_epochs: int
    learning_rate: float

    trainer_cfg: TrainerConfig
    logging_cfg: TrainLoggingConfig
    checkpoint_cfg: TrainCheckpointConfig

@dataclass
class GlobalConfig:
    """
    Scripts GlobalConfig
    """
    dataset: DatasetConfig
    train: TrainConfig
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
