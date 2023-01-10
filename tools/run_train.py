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
from torch.utils.data import DataLoader

from nodetracker.common import conventions
from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets import transforms
from nodetracker.datasets.mot import TorchMOTTrajectoryDataset
from nodetracker.datasets.utils import ode_dataloader_collate_func
from nodetracker.node import load_or_create_model, ModelType
from nodetracker.utils import pipeline

logger = logging.getLogger('TrainScript')


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='train')

    dataset_train_path = os.path.join(cfg.path.assets, cfg.dataset.train_path)
    logger.info(f'Dataset train path: "{dataset_train_path}".')

    postprocess_transform = transforms.transform_factory(cfg.transform.name, cfg.transform.params)
    train_dataset = TorchMOTTrajectoryDataset(
        path=dataset_train_path,
        history_len=cfg.dataset.history_len,
        future_len=cfg.dataset.future_len,
        postprocess=postprocess_transform
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=ode_dataloader_collate_func,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.resources.num_workers,
        shuffle=True
    )

    dataset_val_path = os.path.join(cfg.path.assets, cfg.dataset.val_path)
    logger.info(f'Dataset val path: "{dataset_val_path}".')
    val_dataset = TorchMOTTrajectoryDataset(
        path=dataset_val_path,
        history_len=cfg.dataset.history_len,
        future_len=cfg.dataset.future_len,
        postprocess=postprocess_transform
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        collate_fn=ode_dataloader_collate_func,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.resources.num_workers
    )

    model_type = ModelType.from_str(cfg.model.type)
    assert model_type.trainable, f'Chosen model type "{model_type}" is not trainable!'
    model = load_or_create_model(model_type=model_type, params=cfg.model.params)

    tb_logger = TensorBoardLogger(save_dir=experiment_path, name=conventions.TENSORBOARD_DIRNAME)
    checkpoint_path = conventions.get_checkpoints_dirpath(experiment_path)
    trainer = Trainer(
        gpus=cfg.resources.gpus,
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
        ],
        resume_from_checkpoint=cfg.train.checkpoint_cfg.resume_from if cfg.train.checkpoint_cfg.resume_from
            and os.path.exists(cfg.train.checkpoint_cfg.resume_from) else None
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == '__main__':
    main()
