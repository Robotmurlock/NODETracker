import logging
import os

import hydra
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from nodetracker.common.project import CONFIGS_PATH, ASSETS_PATH, OUTPUTS_PATH
from nodetracker.config_parser import GlobalConfig
from nodetracker.datasets.mot import TorchMOTTrajectoryDataset
from nodetracker.datasets.utils import ode_dataloader_collate_func
from nodetracker.node import LightningODEVAE

logger = logging.getLogger('TrainScript')


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    logger.info(f'Configuration:\n{OmegaConf.to_yaml(cfg)}')
    cfg = GlobalConfig.from_dict(OmegaConf.to_object(cfg))

    dataset_train_path = os.path.join(ASSETS_PATH, cfg.dataset.train_path)
    logger.info(f'Dataset train path: "{dataset_train_path}".')

    train_dataset = TorchMOTTrajectoryDataset(
        path=dataset_train_path,
        history_len=cfg.dataset.history_len,
        future_len=cfg.dataset.future_len
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=ode_dataloader_collate_func,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True
    )

    dataset_val_path = os.path.join(ASSETS_PATH, cfg.dataset.val_path)
    logger.info(f'Dataset val path: "{dataset_val_path}".')
    val_dataset = TorchMOTTrajectoryDataset(
        path=dataset_val_path,
        history_len=cfg.dataset.history_len,
        future_len=cfg.dataset.future_len
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        collate_fn=ode_dataloader_collate_func,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers
    )

    model = LightningODEVAE(
        observable_dim=cfg.model.observable_dim,
        hidden_dim=cfg.model.hidden_dim,
        latent_dim=cfg.model.latent_dim,
        noise_std=cfg.model.train_cfg.noise_std
    )

    tb_logging_path = os.path.join(OUTPUTS_PATH, cfg.train.logging_cfg.path)
    logger.info(f'TensorBoard logger output path: "{tb_logging_path}"')
    tb_logger = TensorBoardLogger(save_dir=tb_logging_path, name=cfg.train.logging_cfg.name)
    trainer = Trainer(
        gpus=cfg.train.trainer_cfg.gpus,
        accelerator=cfg.train.trainer_cfg.accelerator,
        max_epochs=cfg.train.max_epochs,
        logger=tb_logger,
        log_every_n_steps=cfg.train.logging_cfg.log_every_n_steps,
        callbacks=[
            ModelCheckpoint(
                dirpath=cfg.train.checkpoint_cfg.path,
                monitor=cfg.train.checkpoint_cfg.metric_monitor,
                save_last=True,
                save_top_k=1
            )
        ],
        resume_from_checkpoint=cfg.train.checkpoint_cfg.resume_from if cfg.train.checkpoint_cfg.resume_from
            and os.path.exists(cfg.train.checkpoint_cfg.path) else None
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == '__main__':
    main()