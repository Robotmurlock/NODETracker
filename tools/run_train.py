"""
Training script
"""
import logging
import os

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from nodetracker.common import conventions
from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets import transforms
from nodetracker.node import load_or_create_model, ModelType
from nodetracker.utils import pipeline
from tools.utils import create_dataloader

logger = logging.getLogger('TrainScript')


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='train')

    postprocess_transform = transforms.transform_factory(cfg.transform.name, cfg.transform.params)

    # Load train dataset
    train_loader = create_dataloader(
        cfg=cfg,
        split='train',
        transform=postprocess_transform,
        shuffle=True,
        train=True,
        augmentation_before_transform=cfg.augmentations.before_transform,
        augmentation_after_transform=cfg.augmentations.after_transform,
        augmentation_after_batch_collate=cfg.augmentations.after_batch_collate
    )

    # Load val dataset
    val_loader = create_dataloader(
        cfg=cfg,
        split='val',
        transform=postprocess_transform,
        shuffle=False,
        train=False
    )

    model_type = ModelType.from_str(cfg.model.type)
    assert model_type.trainable, f'Chosen model type "{model_type}" is not trainable!'
    model = load_or_create_model(
        model_type=model_type,
        params=cfg.model.params,
        train_params=cfg.train.train_params
    )

    tb_logger = TensorBoardLogger(save_dir=experiment_path, name=conventions.TENSORBOARD_DIRNAME)
    checkpoint_path = conventions.get_checkpoints_dirpath(experiment_path)
    trainer = Trainer(
        devices=cfg.resources.devices,
        accelerator=cfg.resources.accelerator,
        max_epochs=cfg.train.max_epochs,
        logger=tb_logger,
        log_every_n_steps=cfg.train.logging_cfg.log_every_n_steps,
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_path,
                monitor=cfg.train.checkpoint_cfg.metric_monitor,
                save_last=True,
                save_top_k=1
            )
        ]
    )

    if cfg.train.checkpoint_cfg.resume_from:
        assert os.path.exists(cfg.train.checkpoint_cfg.resume_from), f'Failed to find checkpoint "{cfg.train.checkpoint_cfg.resume_from}".'

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.train.checkpoint_cfg.resume_from
    )


if __name__ == '__main__':
    main()
