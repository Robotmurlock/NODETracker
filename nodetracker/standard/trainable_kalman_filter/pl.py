import functools
from typing import Any, Tuple, Optional

import torch

from nodetracker.node.utils.training import LightningModuleBase, LightningTrainConfig
from nodetracker.standard.trainable_kalman_filter.core import TrainingAKFMode, TrainableAdaptiveKalmanFilter
from nodetracker.standard.trainable_kalman_filter.loss import LinearGaussianEnergyFunction


class LightningAdaptiveKalmanFilter(LightningModuleBase):
    def __init__(
        self,
        sigma_p: float = 0.05,
        sigma_p_init_mult: float = 2.0,
        sigma_v: float = 0.00625,
        sigma_v_init_mult: float = 10.0,
        sigma_r: float = 1.0,
        sigma_q: float = 1.0,
        dt: float = 1.0,

        training_mode: str = 'all',
        train_config: Optional[LightningTrainConfig] = None
    ):
        """
        Args:
            sigma_p: Position uncertainty parameter (multiplier)
            sigma_p_init_mult: Position uncertainty parameter for initial cov matrix (multiplier)
            sigma_v: Velocity uncertainty parameter (multiplier)
            sigma_v_init_mult: Velocity uncertainty parameter for initial cov matrix (multiplier)
            sigma_r: Measurement noise multiplier (matrix R)
            sigma_q: Process noise multiplier (matrix Q)
            dt: Step period

            training_mode: Model training mode
        """
        super().__init__(train_config=train_config)
        training_mode = TrainingAKFMode.from_str(training_mode)
        assert training_mode != TrainingAKFMode.FROZEN, 'Can\'t train a model with frozen parameters!'

        self._model = TrainableAdaptiveKalmanFilter(
            sigma_p=sigma_p,
            sigma_p_init_mult=sigma_p_init_mult,
            sigma_v=sigma_v,
            sigma_v_init_mult=sigma_v_init_mult,
            sigma_r=sigma_r,
            sigma_q=sigma_q,
            dt=dt,
            training_mode=training_mode
        )
        self._loss = LinearGaussianEnergyFunction()

    @functools.wraps(TrainableAdaptiveKalmanFilter.initiate)
    def initiate(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._model.initiate(z)

    @functools.wraps(TrainableAdaptiveKalmanFilter.predict)
    def predict(self, x: torch.Tensor, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._model.predict(x, P)

    @functools.wraps(TrainableAdaptiveKalmanFilter.multistep_predict)
    def multistep_predict(self, x: torch.Tensor, P: torch.Tensor, n_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._model.multistep_predict(x, P, n_steps)

    @functools.wraps(TrainableAdaptiveKalmanFilter.update)
    def update(self, x_hat: torch.Tensor, P_hat: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._model.update(x_hat, P_hat, z)

    def forward(self, x: torch.Tensor, P: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        self._model(x, P)

    def forward_loss_step(self, zs: torch.Tensor) -> torch.Tensor:
        n_steps = zs.shape[0]
        assert n_steps >= 2, 'Requires at least 2 steps to train a model!'

        x, P = self._model.initiate(zs[0])
        total_loss = 0.0

        for i in range(1, n_steps):
            z = zs[i]

            # Predict and calculate loss
            x_hat, P_hat = self._model.predict(x, P)
            x_proj, P_proj = self._model.project(x_hat, P_hat)
            innovation = z - x_proj
            total_loss += self._loss(innovation, P_proj)

            # Update
            x, P = self._model.update(x_hat, P_hat, z)

        return total_loss / (n_steps - 1)

    def training_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        zs, zs_unobs, _, _, _ = batch
        zs = torch.cat([zs, zs_unobs], dim=0)
        loss =  self.forward_loss_step(zs)
        self._meter.push('training/loss', loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        zs, zs_unobs, _, _, _ = batch
        zs = torch.cat([zs, zs_unobs], dim=0)
        loss =  self.forward_loss_step(zs)
        self._meter.push('val/loss', loss)
        return loss


def run_takf_test():
    model = LightningAdaptiveKalmanFilter()
    zs = torch.randn(32, 4)
    loss = model.forward_loss_step(zs)


if __name__ == '__main__':
    run_takf_test()
