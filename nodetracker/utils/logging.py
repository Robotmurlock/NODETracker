"""
Logging Utility
"""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, DictConfig

from nodetracker.config_parser import GlobalConfig

logger = logging.getLogger('UtilsLogging')


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configures logger

    Args:
        level: Logging level
    """
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    torch.set_printoptions(sci_mode=False, precision=3)
    logging.basicConfig(
        level=logging.getLevelName(level),
        format='%(asctime)s [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)


def parse_config(cfg: DictConfig) -> Tuple['GlobalConfig', dict]:
    """
    Parses config from DictConfig to GlobalConfig

    Args:
        cfg: parsed DictConfig

    Returns:
        parsed GlobalConfig object, raw (dictionary) config
    """
    logger.info(f'Configuration:\n{OmegaConf.to_yaml(cfg)}')
    raw_cfg = OmegaConf.to_object(cfg)
    cfg = GlobalConfig.from_dict(raw_cfg)

    return cfg, raw_cfg

def save_config(cfg: dict, path: str) -> None:
    """
    Saves config as yaml file

    Args:
        cfg: Raw dict config
        path: Path where to store config
    """
    logger.info(f'Saving config to "{path}"')
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f)

