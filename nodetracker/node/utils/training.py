"""
NODE Lightning module training utility.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import nn

from nodetracker.utils import torch_helper
from nodetracker.utils.meter import MetricMeter


@dataclass
class LightningTrainConfig:
    learning_rate: float = field(default=1e-3)
    sched_lr_gamma: float = field(default=1.0)
    sched_lr_step: int = field(default=1)

    weight_decay: float = field(default=0.0)


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
        optimizer = torch.optim.Adam(
            params=self._model.parameters(),
            lr=self._train_config.learning_rate,
            weight_decay=self._train_config.weight_decay
        )

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


class LightningModuleForecaster(LightningModuleBase):
    """
    Trainer wrapper for RNN like models.
    """
    def __init__(
        self,
        train_config: Optional[LightningTrainConfig],
        model: nn.Module,
        loss_func: nn.Module,
        model_gaussian: bool = False,
    ):
        super().__init__(train_config=train_config)
        self._model_gaussian = model_gaussian
        self._model = model
        self._loss_func = loss_func

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, ...]:
        return self._model(x, t_obs, t_unobs)

    def _calc_loss(self, bboxes_unobs: torch.Tensor, bboxes_unobs_hat: torch.Tensor) -> torch.Tensor:
        """
        Calculates loss based on set loss function.

        Args:
            bboxes_unobs: Ground truth bboxes
            bboxes_unobs_hat: Predicted bboxes

        Returns:
            Loss value
        """
        if self._model_gaussian:
            bboxes_unobs_hat = bboxes_unobs_hat.view(*bboxes_unobs_hat.shape[:-1], -1, 2)
            bboxes_unobs_hat_mean = bboxes_unobs_hat[..., 0]
            bboxes_unobs_hat_log_var = bboxes_unobs_hat[..., 1]
            bboxes_unobs_hat_var = torch.exp(bboxes_unobs_hat_log_var)
            return self._loss_func(bboxes_unobs_hat_mean, bboxes_unobs, bboxes_unobs_hat_var)

        return self._loss_func(bboxes_unobs_hat, bboxes_unobs)

    def training_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, _ = batch
        bboxes_unobs_hat, *_ = self.forward(bboxes_obs, ts_obs, ts_unobs)
        loss = self._calc_loss(bboxes_unobs, bboxes_unobs_hat)

        self._meter.push('training/loss', loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, _ = batch
        bboxes_unobs_hat, *_ = self.forward(bboxes_obs, ts_obs, ts_unobs)
        loss = self._calc_loss(bboxes_unobs, bboxes_unobs_hat)

        self._meter.push('val/loss', loss)
        return loss
