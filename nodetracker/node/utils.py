"""
NODE Lightning module utility class.
"""
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch

from nodetracker.utils import torch_helper
from nodetracker.utils.meter import MetricMeter


@dataclass
class LightningTrainConfig:
    learning_rate: float = field(default=1e-3)
    sched_lr_gamma: float = field(default=1.0)
    sched_lr_step: int = field(default=1)

class LightningModuleBase(pl.LightningModule):
    """
    PytorchLightning module wrapper with some simple default utilities.
    """
    def __init__(self, train_config: LightningTrainConfig):
        super().__init__()
        self._train_config = train_config
        self._meter = MetricMeter()

    def on_validation_epoch_end(self) -> None:
        for name, value in self._meter.get_all():
            self.log(name, value, prog_bar=True)

        # noinspection PyTypeChecker
        optimizer: torch.optim.Optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        lr = torch_helper.get_optim_lr(optimizer)
        self.log('train/lr', lr)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self._model.parameters(), lr=self._train_config.learning_rate)

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=self._train_config.sched_lr_step,
                gamma=self._train_config.sched_lr_gamma
            ),
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]
