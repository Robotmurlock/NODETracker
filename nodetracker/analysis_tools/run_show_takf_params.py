"""
Show TAKF parameters
"""
import logging
import os

import hydra
import torch
from omegaconf import DictConfig

from nodetracker.common import conventions
from nodetracker.common.project import CONFIGS_PATH
from nodetracker.node import load_or_create_model
from nodetracker.standard.trainable_kalman_filter import LightningAdaptiveKalmanFilter
from nodetracker.utils import pipeline

logger = logging.getLogger('ShowTAKFParameters')

torch.set_printoptions(sci_mode=False, precision=5, linewidth=160)


# noinspection PyProtectedMember
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='inference')

    dataset_path = os.path.join(cfg.path.assets, cfg.dataset.get_split_path(cfg.eval.split))
    logger.info(f'Dataset {cfg.eval.split} path: "{dataset_path}".')

    checkpoint_path = conventions.get_checkpoint_path(experiment_path, cfg.eval.checkpoint) \
        if cfg.eval.checkpoint else None
    model = load_or_create_model(
        model_type=cfg.model.type,
        params=cfg.model.params,
        checkpoint_path=checkpoint_path
    )
    assert isinstance(model, LightningAdaptiveKalmanFilter)

    print('MOTION MATRIX:', model._model.get_motion_matrix())
    print('SIGMA P:', model._model._sigma_p)
    print('SIGMA P (init mult):', model._model._sigma_p_init_mult)
    print('SIGMA V:', model._model._sigma_v)
    print('SIGMA V (init mult):', model._model._sigma_v_init_mult)
    print('SIGMA R:', model._model._sigma_r)
    print('SIGMA Q:', model._model._sigma_q)



if __name__ == '__main__':
    main()
