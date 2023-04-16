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

        # Training parameters - not important for inference
        positive_motion_mat: bool = True,
        triu_motion_mat: bool = True,
        first_principles_motion_mat: bool = True,

        training_mode: str = 'all',
        optimize_likelihood: bool = True,
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

            positive_motion_mat: Use positive motion matrix `A >= 0` (non-negative)
            triu_motion_mat: Use upper triangular motion matrix
            first_principles_motion_mat: Use first principles motion matrix as initial parameters

            training_mode: Model training mode
            optimize_likelihood: Optimize negative log likelihood instead of MSE
            train_config: Training configuration
        """
        super().__init__(train_config=train_config)
        training_mode = TrainingAKFMode.from_str(training_mode)
        if train_config is not None:
            assert training_mode != TrainingAKFMode.FROZEN, 'Can\'t train a model with frozen parameters!'
            assert training_mode != TrainingAKFMode.MOTION or not optimize_likelihood, \
                'Can\'t train uncertainty parameters with MSE loss!'


        self._model = TrainableAdaptiveKalmanFilter(
            sigma_p=sigma_p,
            sigma_p_init_mult=sigma_p_init_mult,
            sigma_v=sigma_v,
            sigma_v_init_mult=sigma_v_init_mult,
            sigma_r=sigma_r,
            sigma_q=sigma_q,
            dt=dt,

            training_mode=training_mode,
            positive_motion_mat=positive_motion_mat,
            triu_motion_mat=triu_motion_mat,
            first_principles_motion_mat=first_principles_motion_mat
        )
        self._loss = LinearGaussianEnergyFunction(optimize_likelihood=optimize_likelihood)

    @functools.wraps(TrainableAdaptiveKalmanFilter.initiate)
    def initiate(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._model.initiate(z)

    @functools.wraps(TrainableAdaptiveKalmanFilter.predict)
    def predict(self, x: torch.Tensor, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._model.predict(x, P)

    @functools.wraps(TrainableAdaptiveKalmanFilter.multistep_predict)
    def multistep_predict(self, x: torch.Tensor, P: torch.Tensor, n_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._model.multistep_predict(x, P, n_steps)

    def inference(self, zs: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, ...]:
        """
        Inference interface.

        Args:
            zs: Trajectory history
            t_obs: Observed trajectory time points (not used) - interface consistency
            t_unobs: Unobserved trajectory time points -> inferred trajectory length

        Returns:
            Trajectory prediction for N steps ahead.
        """
        _ = t_obs  # unused
        n_steps: int = t_unobs.shape[0]

        x, P = self._model.initiate(zs[0])
        return self.multistep_predict(x, P, n_steps)

    @functools.wraps(TrainableAdaptiveKalmanFilter.update)
    def update(self, x_hat: torch.Tensor, P_hat: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._model.update(x_hat, P_hat, z)

    def forward(self, x: torch.Tensor, P: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        self._model(x, P)

    def forward_loss_step(self, zs_obs: torch.Tensor, zs_unobs: torch.Tensor) -> torch.Tensor:
        zs = torch.cat([zs_obs, zs_unobs], dim=0)
        n_warmup_steps = zs_obs.shape[0]
        n_steps = zs.shape[0]
        assert n_steps >= 2, 'Requires at least 2 steps to train a model!'

        x, P = self._model.initiate(zs[0])
        total_loss = 0.0

        for i in range(1, n_steps):
            z = zs[i]
            z_expanded = z.unsqueeze(-1)

            # Predict and calculate loss
            x_hat, P_hat = self._model.predict(x, P)
            x_proj, P_proj = self._model.project(x_hat, P_hat, flatten=False)
            innovation = z_expanded - x_proj

            if i >= n_warmup_steps:
                total_loss += self._loss(innovation, P_proj)

            # Update
            x, P = self._model.update(x_hat, P_hat, z)

        return total_loss / (n_steps - 1)

    def training_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        zs_obs, zs_unobs, _, _, _ = batch
        loss =  self.forward_loss_step(zs_obs, zs_unobs)
        self._meter.push('training/loss', loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        zs_obs, zs_unobs, _, _, _ = batch
        loss =  self.forward_loss_step(zs_obs, zs_unobs)
        self._meter.push('val/loss', loss)
        return loss


def run_takf_test():
    model = LightningAdaptiveKalmanFilter()
    zs = torch.randn(10, 32, 4)
    zs_unobs = torch.randn(1, 32, 4)
    _ = model.forward_loss_step(zs, zs_unobs)


if __name__ == '__main__':
    run_takf_test()
