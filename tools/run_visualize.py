"""
Training script
"""
import logging
import os

import hydra
from omegaconf import DictConfig

from nodetracker.common.project import CONFIGS_PATH
from nodetracker.utils import pipeline

logger = logging.getLogger('VizScript')


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='visualize')
    assert cfg.visualize is not None, 'Visualize config are not defined!'

    predictions_path = os.path.join(cfg.path.master, cfg.visualize.predictions_path)


if __name__ == '__main__':
    main()
