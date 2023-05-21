"""
Pipeline utils (tools scripts)
"""
import logging
from typing import Tuple, Optional, Any

from omegaconf import DictConfig
from omegaconf import OmegaConf

from nodetracker.common import conventions
from nodetracker.config_parser import GlobalConfig

logger = logging.getLogger('PipelineUtils')


def preprocess(cfg: DictConfig, name: str, cls: Optional[Any] = None) -> Tuple[GlobalConfig, str]:
    """
    Pipeline preprocess:
    - Parse GlobalConfig from DictConfig (hydra)
    - Show parsed data
    - Save config in experiment path

    Args:
        cfg: Hydra parsed pipeline config
        name: Pipeline (script) name
        cls: ConfigClass (optional) - GlobalConfig extension

    Returns:
        GlobalConfig and model experiment path
    """
    if cls is None:
        cls = GlobalConfig
    assert issubclass(cls, GlobalConfig), f'Expected extension of GlobalConfig but found "{cls.__name__}"!'

    raw_cfg = OmegaConf.to_object(cfg)

    cfg = cls.from_dict(raw_cfg)
    logger.info(f'Config:\n{cfg.prettyprint}')

    experiment_path = conventions.get_experiment_path(cfg.path.master, cfg.dataset.name, cfg.train.experiment)
    logger.info(f'Experiment output path: "{experiment_path}"')
    cfg.save(conventions.get_config_path(experiment_path, f'{name}.yaml'))
    return cfg, experiment_path
