"""
Pipeline utils (tools scripts)
"""
import logging
from typing import Tuple

from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.utils import instantiate

from nodetracker.common import conventions
from nodetracker.config_parser import GlobalConfig
from nodetracker.datasets.augmentations import CompositionAugmentation

logger = logging.getLogger('PipelineUtils')


def preprocess(cfg: DictConfig, name: str) -> Tuple[GlobalConfig, str]:
    """
    Pipeline preprocess:
    - Parse GlobalConfig from DictConfig (hydra)
    - Show parsed data
    - Save config in experiment path

    Args:
        cfg: Hydra parsed pipeline config
        name: Pipeline (script) name

    Returns:
        GlobalConfig and model experiment path
    """
    raw_cfg = OmegaConf.to_object(cfg)
    if 'augmentations' in raw_cfg:
        # noinspection PyTypeChecker
        raw_cfg.augmentations.before_transform = CompositionAugmentation([instantiate(a) for a in raw_cfg.augmentations.before_transform])
        # noinspection PyTypeChecker
        raw_cfg.augmentations.after_transform = CompositionAugmentation([instantiate(a) for a in raw_cfg.augmentations.after_transform])

    cfg = GlobalConfig.from_dict(raw_cfg)
    logger.info(f'Config:\n{cfg.prettyprint}')

    experiment_path = conventions.get_experiment_path(cfg.path.master, cfg.dataset.name, cfg.train.experiment)
    logger.info(f'Experiment output path: "{experiment_path}"')
    cfg.save(conventions.get_config_path(experiment_path, f'{name}.yaml'))
    return cfg, experiment_path
