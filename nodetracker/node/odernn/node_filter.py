from typing import Optional, Tuple, Union, Dict

import torch
from torch import nn

from nodetracker.datasets.transforms import InvertibleTransformWithVariance, InvertibleTransform
from nodetracker.evaluation.metrics.sot import metrics_func
from nodetracker.library.building_blocks.mlp import MLP
from nodetracker.node.core.odevae import MLPODEF, NeuralODE
from nodetracker.node.core.solver.factory import ode_solver_factory
from nodetracker.node.losses.factory import factory_loss_function
from nodetracker.node.utils.training import LightningTrainConfig, LightningModuleBase


class NODEFilterModel(nn.Module):
    """
    NODEFilterModel
    """
    def __init__(
        self,
        observable_dim: int,
        latent_dim: int,
        output_dim: int,

        model_gaussian: bool = False,

        n_ode_mlp_layers: int = 2,
        n_update_mlp_layers: int = 2,
        n_head_mlp_layers: int = 2,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None
    ):
        super().__init__()

        self._observable_dim = observable_dim
        self._latent_dim = latent_dim
        self._output_dim = output_dim

        func = MLPODEF(latent_dim, latent_dim, n_layers=n_ode_mlp_layers)
        solver = ode_solver_factory(solver_name, solver_params)
        self._ode = NeuralODE(func=func, solver=solver)

        self._prior_head = nn.Sequential(
            MLP(latent_dim, latent_dim, n_layers=n_head_mlp_layers),
            nn.Linear(latent_dim, 2 * output_dim if model_gaussian else output_dim, bias=True)
        )
        self._posterior_head = nn.Sequential(
            MLP(latent_dim, latent_dim, n_layers=n_head_mlp_layers),
            nn.Linear(latent_dim, 2 * output_dim if model_gaussian else output_dim, bias=True)
        )

        self._obs2latent = MLP(
            input_dim=observable_dim,
            hidden_dim=latent_dim,
            output_dim=latent_dim
        )
        self._unobs2latent = MLP(
            input_dim=output_dim,
            hidden_dim=latent_dim,
            output_dim=latent_dim
        )
        self._update_layer = MLP(
            input_dim=2 * latent_dim,
            hidden_dim=latent_dim,
            output_dim=latent_dim,
            n_layers=n_update_mlp_layers
        )

    def initialize(self, batch_size: int, device: Union[str, torch.device]) -> Tuple[torch.Tensor, torch.Tensor]:
        z0 = torch.zeros(batch_size, self._latent_dim, dtype=torch.float32).to(device)
        t0 = torch.zeros(batch_size, 1, dtype=torch.float32).to(device)
        return z0, t0

    def estimate_prior(self, z0: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ts = torch.stack([t0, t1])
        z1_prior = self._ode(z0, ts)
        x1_prior = self._prior_head(z1_prior)
        return x1_prior, z1_prior

    def estimate_posterior(self, z1_prior: torch.Tensor, z1_evidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z1_combined = torch.cat([z1_prior, z1_evidence], dim=-1)
        z1_posterior = z1_prior + self._update_layer(z1_combined)
        x1_posterior = self._posterior_head(z1_posterior)
        return x1_posterior, z1_posterior

    def next(self, z0: torch.Tensor, z1_evidence: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1_prior, z1_prior = self.estimate_prior(z0, t0, t1)
        x1_posterior, z1_posterior = self.estimate_posterior(z1_prior, z1_evidence)
        return x1_prior, x1_posterior, z1_posterior

    def encode_obs(self, x: torch.Tensor) -> torch.Tensor:
        return self._obs2latent(x)

    def encode_unobs(self, x: torch.Tensor) -> torch.Tensor:
        return self._unobs2latent(x)

    def encode_obs_trajectory(self, x_obs: torch.Tensor, t_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_warmup_steps, batch_size, _ = t_obs.shape

        z0, t0 = self.initialize(batch_size, t_obs.device)
        z1_obs = self.encode_obs(x_obs)
        for i in range(n_warmup_steps):
            t1 = t_obs[i]
            z1 = z1_obs[i]
            _, _, z0 = self.next(z0, z1, t0, t1)
            t0 = t1

        return z0, t0

    def forward(self, x_obs: torch.Tensor, t_obs: torch.Tensor, x_unobs: torch.Tensor, t_unobs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        z0, t0 = self.encode_obs_trajectory(x_obs, t_obs)

        n_estimation_steps, _, _ = t_unobs.shape
        priors, posteriors = [], []
        z1_unobs = self.encode_unobs(x_unobs)
        for i in range(n_estimation_steps):
            z1 = z1_unobs[i]
            t1 = t_unobs[i]
            x1_prior, x1_posterior, z0 = self.next(z0, z1, t0, t1)
            t0 = t1

            priors.append(x1_prior)
            posteriors.append(x1_posterior)

        prior = torch.stack(priors)
        posterior = torch.stack(posteriors)

        return prior, posterior


class LightningNODEFilterModel(LightningModuleBase):
    """
    PytorchLightning wrapper for NODEFilterModel model.
    """
    def __init__(
        self,
        observable_dim: int,
        latent_dim: int,
        output_dim: Optional[int] = None,

        model_gaussian: bool = False,
        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None,

        n_ode_mlp_layers: int = 2,
        n_update_mlp_layers: int = 2,
        n_head_mlp_layers: int = 2,

        solver_name: Optional[str] = None,
        solver_params: Optional[dict] = None,

        train_config: Optional[LightningTrainConfig] = None,
        log_epoch_metrics: bool = True
    ):
        if output_dim is None:
            output_dim = observable_dim

        super().__init__(train_config=train_config)

        self._model = NODEFilterModel(
            observable_dim=observable_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            model_gaussian=model_gaussian,
            solver_name=solver_name,
            solver_params=solver_params,
            n_ode_mlp_layers=n_ode_mlp_layers,
            n_update_mlp_layers=n_update_mlp_layers,
            n_head_mlp_layers=n_head_mlp_layers
        )

        self._model_gaussian = model_gaussian
        self._loss_func = factory_loss_function(train_config.loss_name, train_config.loss_params) \
            if train_config is not None else None
        if self._model_gaussian:
            if train_config is not None:
                assert 'gaussian_nllloss' in train_config.loss_name, \
                    'Failed to find "gaussian_nllloss" in loss function name!'
            if transform_func is not None:
                assert isinstance(transform_func, InvertibleTransformWithVariance), \
                    f'Expected transform function to be of type "InvertibleTransformWithStd" ' \
                    f'but got "{type(transform_func)}"'
        self._transform_func = transform_func
        self._log_epoch_metrics = log_epoch_metrics

    @property
    def core(self) -> NODEFilterModel:
        return self._model

    def forward(self, x_obs: torch.Tensor, t_obs: torch.Tensor, x_unobs: torch.Tensor, t_unobs: torch.Tensor, *args, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        return self._model(x_obs, t_obs, x_unobs, t_unobs, *args, **kwargs)

    def inference(self, x_obs: torch.Tensor, t_obs: torch.Tensor, x_unobs: torch.Tensor, t_unobs: torch.Tensor, *args, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference (alias for forward)

        Args:
            x_obs: Observed data
            t_obs: Observed time points
            x_unobs: Unobserved data
            t_unobs: Unobserved time points

        Returns:
            Prior and posterior estimation for future trajectory
        """
        return self._model(x_obs, t_obs, x_unobs, t_unobs, *args, **kwargs)

    @staticmethod
    def _raw_to_mean_var(bboxes_unobs_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bboxes_unobs_hat = bboxes_unobs_hat.view(*bboxes_unobs_hat.shape[:-1], -1, 2)
        bboxes_unobs_hat_mean = bboxes_unobs_hat[..., 0]
        bboxes_unobs_hat_log_var = bboxes_unobs_hat[..., 1]
        bboxes_unobs_hat_var = torch.exp(bboxes_unobs_hat_log_var)
        return bboxes_unobs_hat_mean, bboxes_unobs_hat_var

    def _calc_loss_and_metrics(
        self,
        bboxes_obs: torch.Tensor,
        original_bboxes_unobs: torch.Tensor,
        transformed_bboxes_unobs: torch.Tensor,
        bboxes_unobs_prior: torch.Tensor,
        bboxes_unobs_posterior: torch.Tensor,
        metadata: dict
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Optional[Dict[str, float]]]:
        """
        Calculates loss based on set loss function.

        Args:
            bboxes_obs: Observed bboxes (original)
            original_bboxes_unobs: Original (not transformed) Ground truth bboxes
            transformed_bboxes_unobs: Original (transformed) Ground truth bboxes
            bboxes_unobs_prior: Estimated prior bboxes
            bboxes_unobs_posterior: Estimated posterior bboxes

        Returns:
            Loss value (or mapping)
        """
        if self._model_gaussian:
            bboxes_unobs_prior_mean, bboxes_unobs_prior_var = self._raw_to_mean_var(bboxes_unobs_prior)
            bboxes_unobs_posterior_mean, bboxes_unobs_posterior_var = self._raw_to_mean_var(bboxes_unobs_posterior)

            prior_loss = self._loss_func(bboxes_unobs_prior_mean, transformed_bboxes_unobs, bboxes_unobs_prior_var)
            posterior_loss = self._loss_func(bboxes_unobs_posterior_mean, transformed_bboxes_unobs, bboxes_unobs_posterior_var)
            loss = {
                'prior_loss': prior_loss,
                'posterior_loss': posterior_loss,
                'loss': prior_loss + posterior_loss
            }

            if self._transform_func is not None:
                # Invert mean
                _, bboxes_unobs_prior_mean, *_ = self._transform_func.inverse([bboxes_obs, bboxes_unobs_prior_mean, metadata, None], shallow=False)
                _, bboxes_unobs_posterior_mean, *_ = self._transform_func.inverse([bboxes_obs, bboxes_unobs_posterior_mean, metadata, None],
                                                                                  shallow=False)

            if not self._log_epoch_metrics:
                return loss, None

            gt_traj = original_bboxes_unobs.detach().cpu().numpy()
            prior_traj = bboxes_unobs_prior_mean.detach().cpu().numpy()
            prior_metrics = metrics_func(gt_traj, prior_traj)
            prior_metrics = {f'prior_{name}': value for name, value in prior_metrics.items()}

            gt_traj = original_bboxes_unobs.detach().cpu().numpy()
            posterior_traj = bboxes_unobs_posterior_mean.detach().cpu().numpy()
            posterior_metrics = metrics_func(gt_traj, posterior_traj) if self._log_epoch_metrics else None
            posterior_metrics = {f'posterior_{name}': value for name, value in posterior_metrics.items()}
            metrics = dict(list(prior_metrics.items()) + list(posterior_metrics.items()))

            return loss, metrics

        prior_loss = self._loss_func(bboxes_unobs_prior, transformed_bboxes_unobs)
        posterior_loss = self._loss_func(bboxes_unobs_posterior, transformed_bboxes_unobs)
        if isinstance(prior_loss, dict) and isinstance(posterior_loss, dict):
            prior_loss = {f'prior_{name}': value for name, value in prior_loss.items()}
            posterior_loss = {f'prior_{name}': value for name, value in posterior_loss.items()}
            loss = dict(list(prior_loss.items()) + list(posterior_loss.items()))
        else:
            loss = {
                'prior_loss': prior_loss,
                'posterior_loss': posterior_loss
            }

        if self._transform_func is not None:
            _, bboxes_unobs_prior, *_ = self._transform_func.inverse([bboxes_obs, bboxes_unobs_prior, metadata, None], shallow=False)
            _, bboxes_unobs_posterior, *_ = self._transform_func.inverse([bboxes_obs, bboxes_unobs_posterior, metadata, None], shallow=False)

        if not self._log_epoch_metrics:
            return loss, None

        gt_traj = original_bboxes_unobs.detach().cpu().numpy()
        prior_traj = bboxes_unobs_prior.detach().cpu().numpy()
        prior_metrics = metrics_func(gt_traj, prior_traj)
        prior_metrics = {f'prior_{name}': value for name, value in prior_metrics.items()}

        gt_traj = original_bboxes_unobs.detach().cpu().numpy()
        posterior_traj = bboxes_unobs_posterior.detach().cpu().numpy()
        posterior_metrics = metrics_func(gt_traj, posterior_traj) if self._log_epoch_metrics else None
        posterior_metrics = {f'prior_{name}': value for name, value in posterior_metrics.items()}
        metrics = dict(list(prior_metrics.items()) + list(posterior_metrics.items()))

        return loss, metrics

    def _log_loss(self, loss: Union[torch.Tensor, Dict[str, torch.Tensor]], prefix: str, log_step: bool = True) -> None:
        """
        Helper function to log loss. Options:
        - Single value: logged as "{prefix}/loss"
        - Dictionary: for each key log value as "{prefix}/{key}"

        Args:
            loss: Loss
            prefix: Prefix (train or val)
        """
        assert prefix in ['training', 'val'], f'Invalid prefix value "{prefix}"!'

        if isinstance(loss, dict):
            assert 'loss' in loss, \
                f'When returning loss as dictionary it has to have key "loss". Found: {list(loss.keys())}'
            for name, value in loss.items():
                value = value.detach().cpu()
                assert not torch.isnan(value).any(), f'Got nan value for key "{name}"!'
                self._meter.push(f'{prefix}-epoch/{name}', value)
                if log_step:
                    self.log(f'{prefix}/{name}', value, prog_bar=False)
        else:
            loss = loss.detach().cpu()
            assert not torch.isnan(loss).any(), 'Got nan value!'
            loss = loss.detach().cpu()
            self._meter.push(f'{prefix}-epoch/loss', loss)
            if log_step:
                self.log(f'{prefix}/loss', loss, prog_bar=False)

    def _log_metrics(self, metrics: Optional[Dict[str, float]], prefix: str) -> None:
        """
        Helper function to log metrics. Input format:
        - Dictionary: for each key log value as "{prefix}-epoch/{key}"

        Args:
            prefix: Prefix (train or val)
        """
        if metrics is None:
            return

        assert prefix in ['training', 'val'], f'Invalid prefix value "{prefix}"!'
        for name, value in metrics.items():
            self._meter.push(f'{prefix}-metrics/{name}', value)

    def training_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_aug_unobs, ts_obs, ts_unobs, orig_bboxes_obs, orig_bboxes_unobs, bboxes_unobs, metadata = batch
        bboxes_prior, bboxes_posterior = self.forward(bboxes_obs, ts_obs, bboxes_aug_unobs, ts_unobs)

        # noinspection PyTypeChecker
        loss, metrics = self._calc_loss_and_metrics(orig_bboxes_obs, orig_bboxes_unobs, bboxes_unobs, bboxes_prior, bboxes_posterior, metadata)
        self._log_loss(loss, prefix='training', log_step=True)
        self._log_metrics(metrics, prefix='training')

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_aug_unobs, ts_obs, ts_unobs, orig_bboxes_obs, orig_bboxes_unobs, bboxes_unobs, metadata = batch
        bboxes_prior, bboxes_posterior = self.forward(bboxes_obs, ts_obs, bboxes_aug_unobs, ts_unobs)

        # noinspection PyTypeChecker
        loss, metrics = self._calc_loss_and_metrics(orig_bboxes_obs, orig_bboxes_unobs, bboxes_unobs, bboxes_prior, bboxes_posterior, metadata)
        self._log_loss(loss, prefix='val', log_step=False)
        self._log_metrics(metrics, prefix='val')

        return loss
