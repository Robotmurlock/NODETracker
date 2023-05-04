"""
NODE Lightning module training utility.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from nodetracker.node.losses import factory_loss_function
from nodetracker.utils import torch_helper
from nodetracker.utils.meter import MetricMeter
from nodetracker.datasets.transforms import InvertibleTransformWithStd, InvertibleTransform


@dataclass
class LightningTrainConfig:
    loss_name: str = field(default='mse')
    loss_params: dict = field(default_factory=dict)

    learning_rate: float = field(default=1e-3)
    sched_lr_gamma: float = field(default=1.0)
    sched_lr_step: int = field(default=1)

    optim_name: str = field(default='default')
    optim_additional_params: dict = field(default_factory=dict)

    weight_decay: float = field(default=0.0)


class LightningModuleBase(pl.LightningModule):
    """
    PytorchLightning module wrapper with some simple default utilities.
    """
    def __init__(self, train_config: LightningTrainConfig):
        """
        Args:
            train_config: Universal training config
        """
        super().__init__()
        self._train_config = train_config
        self._meter = MetricMeter()

    @property
    def n_params(self) -> int:
        """
        Gets number of model parameters.

        Returns:
            Return number of model parameters
        """
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in trainable_parameters])

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
        optim_name = self._train_config.optim_name.lower()
        optim_catalog = {
            'default': torch.optim.Adam,
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW
        }
        optim_cls = optim_catalog[optim_name]

        optimizer = optim_cls(
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
    Trainer wrapper for RNN like models (NODE-friendly).
    """
    def __init__(
        self,
        train_config: Optional[LightningTrainConfig],
        model: nn.Module,
        model_gaussian: bool = False,
        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithStd]] = None
    ):
        """
        Args:
            train_config: Universal training config
            model: Torch model to train
            model_gaussian: Model Gaussian
            transform_func: Transform function to apply inverse transformation.
        """
        super().__init__(train_config=train_config)
        self._model_gaussian = model_gaussian
        self._model = model
        self._loss_func = factory_loss_function(train_config.loss_name, train_config.loss_params) \
            if train_config is not None else None
        if self._model_gaussian:
            if train_config is not None:
                assert 'gaussian_nllloss' in train_config.loss_name, 'Failed to find "gaussian_nllloss" in loss function name!'
            if transform_func is not None:
                assert isinstance(transform_func, InvertibleTransformWithStd), \
                    f'Expected transform function to be of type "InvertibleTransformWithStd" but got "{type(transform_func)}"'
        self._transform_func = transform_func

    def forward(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None, *args, **kwargs) \
            -> Tuple[torch.Tensor, ...]:
        return self._model(x, t_obs, t_unobs, *args, **kwargs)

    def inference(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, ...]:
        """
        By default, this function is synonym for `forward` but optionally it can be overriden for preprocessing or postprocessing.

        Args:
            x: Trajectory bboxes points
            t_obs: Observable time points
            t_unobs: Unobservable time points

        Returns:
            Model inference (output)
        """
        return self._model(x, t_obs, t_unobs)

    def _calc_loss(self, bboxes_obs: torch.Tensor, bboxes_unobs: torch.Tensor, bboxes_unobs_hat: torch.Tensor) -> torch.Tensor:
        """
        Calculates loss based on set loss function.

        Args:
            bboxes_obs: Observed bboxes
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

            if self._transform_func is not None:
                # Invert mean
                _, bboxes_unobs_hat_mean, *_ = self._transform_func.inverse([bboxes_obs, bboxes_unobs_hat_mean], shallow=False)
                _, bboxes_unobs, *_ = self._transform_func.inverse([bboxes_obs, bboxes_unobs], shallow=False)

                # Invert std
                bboxes_unobs_hat_std = torch.sqrt(bboxes_unobs_hat_var)
                bboxes_unobs_hat_std = self._transform_func.inverse_std(bboxes_unobs_hat_std)
                bboxes_unobs_hat_var = torch.square(bboxes_unobs_hat_std)

            return self._loss_func(bboxes_unobs_hat_mean, bboxes_unobs, bboxes_unobs_hat_var)

        if self._transform_func is not None:
            _, bboxes_unobs_hat, *_ = self._transform_func.inverse([bboxes_obs, bboxes_unobs_hat], shallow=False)
            _, bboxes_unobs, *_ = self._transform_func.inverse([bboxes_obs, bboxes_unobs], shallow=False)

        return self._loss_func(bboxes_unobs_hat, bboxes_unobs)

    def _log_loss(self, loss: Union[torch.Tensor, Dict[str, torch.Tensor]], prefix: str) -> None:
        """
        Helper function to log loss. Options:
        - Single value: logged as "{prefix}/loss"
        - Dictionary: for each key log value as "{prefix}/{key}"

        Args:
            loss: Loss
            prefix: Prefix (train or val)

        Returns:

        """
        assert prefix in ['training', 'val'], f'Invalid prefix value "{prefix}"!'

        if isinstance(loss, dict):
            assert 'loss' in loss, \
                f'When returning loss as dictionary it has to have key "loss". Found: {list(loss.keys())}'
            for name, value in loss.items():
                assert not torch.isnan(value).any(), f'Got nan value for key "{name}"!'
                self._meter.push(f'{prefix}/{name}', value)
        else:
            assert not torch.isnan(loss).any(), f'Got nan value!'
            self._meter.push(f'{prefix}/loss', loss)

    def training_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, _ = batch
        bboxes_unobs_hat, *_ = self.forward(bboxes_obs, ts_obs, ts_unobs)
        loss = self._calc_loss(bboxes_obs, bboxes_unobs, bboxes_unobs_hat)
        self._log_loss(loss, prefix='training')
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, _ = batch
        bboxes_unobs_hat, *_ = self.forward(bboxes_obs, ts_obs, ts_unobs)
        loss = self._calc_loss(bboxes_obs, bboxes_unobs, bboxes_unobs_hat)
        self._log_loss(loss, prefix='val')
        return loss



class LightningModuleForecasterWithTeacherForcing(LightningModuleForecaster):
    """
    Extension of `LightningModuleForecaster` that supports Teacher's forcing.
    """
    def __init__(
        self,
        train_config: Optional[LightningTrainConfig],
        model: nn.Module,
        model_gaussian: bool = False,
        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithStd]] = None,
        teacher_forcing: bool = False
    ):
        """
        Args:
            train_config: Universal training config
            model: Torch model to train
            model_gaussian: Model Gaussian
            teacher_forcing: Apply Teacher forcing method
        """
        super().__init__(train_config=train_config, model=model, model_gaussian=model_gaussian, transform_func=transform_func)
        self._teacher_forcing = teacher_forcing

    def training_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, _ = batch
        bboxes_unobs_hat, *_ = self.forward(bboxes_obs, ts_obs, ts_unobs, x_tf=bboxes_unobs if self._teacher_forcing else None)
        loss = self._calc_loss(bboxes_obs, bboxes_unobs, bboxes_unobs_hat)

        self._meter.push('training/loss', loss)
        return loss
