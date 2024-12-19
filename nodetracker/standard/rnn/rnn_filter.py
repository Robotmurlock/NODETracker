import random
from typing import Optional, Tuple, Union, Dict, List

import torch
from torch import nn

from nodetracker.datasets.augmentations.trajectory import remove_points
from nodetracker.datasets.transforms import InvertibleTransformWithVariance, InvertibleTransform
from nodetracker.evaluation.metrics.sot import metrics_func
from nodetracker.library.building_blocks.mlp import MLP
from nodetracker.node.losses.factory import factory_loss_function
from nodetracker.node.utils.training import LightningTrainConfig, LightningModuleBase


class RNNMultiStepModule(nn.Module):
    """
    RNNMultiStepModule
    """
    def __init__(
        self,
        latent_dim: int,
        n_rnn_layers: int = 1,
        scale: float = 5.0
    ):
        """
        Args:
            latent_dim: "latent" (hidden-2) trajectory dimension
            n_rnn_layers: Number of stacked RNN (GRU) layers
        """
        super().__init__()

        self._latent_dim = latent_dim
        self._n_rnn_layers = n_rnn_layers
        self._scale = scale

        self._rnn = nn.GRU(
            input_size=1,
            hidden_size=latent_dim,
            num_layers=n_rnn_layers,
            batch_first=False
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        n_steps = round(float(t[0, 0, 0]))

        z = z.unsqueeze(0)
        for t_i in range(1, n_steps + 1):
            t_i = torch.tensor([t_i], dtype=torch.float32).view(1, 1, 1).repeat(1, z.shape[1], 1).to(z)
            _, z = self._rnn(t_i, z)

        return z[-1]


class RNNFilterModel(nn.Module):
    """
    RNNFilterModel
    """
    def __init__(
        self,
        observable_dim: int,
        latent_dim: int,
        output_dim: int,

        homogeneous: bool = False,  # Observed and unobserved data points are processes the same way
        bounded_variance: bool = False,
        rnn_update_layer: bool = False,

        n_rnn_layers: int = 1,
        n_update_layers: int = 1,
        n_head_mlp_layers: int = 2,
        n_obs2latent_mlp_layers: int = 1,
        scale: float = 5.0
    ):
        super().__init__()
        self._bounded_variance = bounded_variance
        self._rnn_update_layer = rnn_update_layer

        self._observable_dim = observable_dim
        self._latent_dim = latent_dim
        self._output_dim = output_dim

        self._rnn = RNNMultiStepModule(
            latent_dim=latent_dim,
            n_rnn_layers=n_rnn_layers,
            scale=scale
        )

        self._prior_head_mean = nn.Sequential(
            MLP(latent_dim, latent_dim, n_layers=n_head_mlp_layers),
            nn.Linear(latent_dim, output_dim, bias=True)
        )
        self._prior_head_log_var = nn.Sequential(
            MLP(latent_dim, latent_dim, n_layers=n_head_mlp_layers),
            nn.Linear(latent_dim, output_dim, bias=True)
        )
        self._posterior_head_mean = nn.Sequential(
            MLP(latent_dim, latent_dim, n_layers=n_head_mlp_layers),
            nn.Linear(latent_dim, output_dim, bias=True)
        )
        self._posterior_head_log_var = nn.Sequential(
            MLP(latent_dim, latent_dim, n_layers=n_head_mlp_layers),
            nn.Linear(latent_dim, output_dim, bias=True)
        )

        self._obs2latent = MLP(
            input_dim=observable_dim,
            hidden_dim=latent_dim,
            output_dim=latent_dim,
            n_layers=n_obs2latent_mlp_layers
        )

        if homogeneous:
            assert observable_dim == output_dim, 'For homogeneous model, observable and output dimension must match!'
            self._unobs2latent = self._obs2latent  # Alias
        else:
            self._unobs2latent = MLP(
                input_dim=output_dim,
                hidden_dim=latent_dim,
                output_dim=latent_dim,
                n_layers=n_obs2latent_mlp_layers
            )

        self._n_update_layers = n_update_layers
        if rnn_update_layer:
            self._update_layer = nn.GRU(
                input_size=latent_dim,
                hidden_size=latent_dim,
                num_layers=n_update_layers,
                batch_first=False
            )
        else:
            self._update_layer = MLP(
                input_dim=2 * latent_dim,
                hidden_dim=latent_dim,
                output_dim=latent_dim,
                n_layers=n_update_layers
            )

    def initialize(self, batch_size: int, device: Union[str, torch.device]) -> Tuple[torch.Tensor, torch.Tensor]:
        z0 = torch.zeros(batch_size, self._latent_dim, dtype=torch.float32).to(device)
        t0 = torch.zeros(batch_size, 1, dtype=torch.float32).to(device)
        return z0, t0

    def estimate_prior(self, z0: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = (t1 - t0).unsqueeze(0)
        z1_prior = self._rnn(z0, t)
        x1_prior_mean = self._prior_head_mean(z1_prior)
        x1_prior_log_var = self._prior_head_log_var(z1_prior)
        return x1_prior_mean, x1_prior_log_var, z1_prior

    def estimate_posterior(self, z1_prior: torch.Tensor, z1_evidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._rnn_update_layer:
            z1_prior = z1_prior.unsqueeze(0).repeat(self._n_update_layers, 1, 1)
            z1_evidence = z1_evidence.unsqueeze(0)
            z1_posterior, _ = self._update_layer(z1_evidence, z1_prior)
            z1_posterior = z1_posterior[0]
        else:
            z1_combined = torch.cat([z1_prior, z1_evidence], dim=-1)
            z1_posterior = self._update_layer(z1_combined)

        x1_posterior_mean = self._posterior_head_mean(z1_posterior)
        x1_posterior_log_var = self._posterior_head_log_var(z1_posterior)
        return x1_posterior_mean, x1_posterior_log_var, z1_posterior

    def next(self, z0: torch.Tensor, z1_evidence: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1_prior_mean, x1_prior_log_var, z1_prior = self.estimate_prior(z0, t0, t1)
        x1_posterior_mean, x1_posterior_log_var, z1_posterior = self.estimate_posterior(z1_prior, z1_evidence)
        return x1_prior_mean, x1_prior_log_var, x1_posterior_mean, x1_posterior_log_var, z1_posterior

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
            _, _, _, _, z0 = self.next(z0, z1, t0, t1)
            t0 = t1

        return z0, t0

    def forward(self, x_obs: torch.Tensor, t_obs: torch.Tensor, x_unobs: torch.Tensor, t_unobs: torch.Tensor, mask: Optional[List[bool]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z0, t0 = self.encode_obs_trajectory(x_obs, t_obs)

        n_estimation_steps, _, _ = t_unobs.shape
        priors_mean, priors_log_var, posteriors_mean, posteriors_log_var = [], [], [], []
        z1_unobs = self.encode_unobs(x_unobs)
        i_lag = 0  # measurement index lags in case of missing points
        for i in range(n_estimation_steps):
            t1 = t_unobs[i]
            m1 = mask[i] if mask is not None else True
            if m1:
                # Data is present - Estimate the prior and the posterior
                z1 = z1_unobs[i - i_lag]
                x1_prior_mean, x1_prior_log_var, x1_posterior_mean, x1_posterior_log_var, z0 = self.next(z0, z1, t0, t1)
                posteriors_mean.append(x1_posterior_mean)
                posteriors_log_var.append(x1_posterior_log_var)
            else:
                # Data is not present - Just estimate the prior
                x1_prior_mean, x1_prior_log_var, z0 = self.estimate_prior(z0, t0, t1)
                i_lag += 1

            t0 = t1

            priors_mean.append(x1_prior_mean)
            priors_log_var.append(x1_prior_log_var)

        priors_mean, priors_log_var, posteriors_mean, posteriors_log_var = \
            [(torch.stack(v) if len(v) > 0 else torch.empty(0, dtype=torch.float32))
             for v in [priors_mean, priors_log_var, posteriors_mean, posteriors_log_var]]

        assert z0 is not None
        return priors_mean, priors_log_var, posteriors_mean, posteriors_log_var, z0

    def postprocess_log_var(self, x: torch.Tensor) -> torch.Tensor:
        if not self._bounded_variance:
            return torch.exp(x)
        else:
            return 0.1 + 0.9 * torch.nn.functional.softplus(x)


class LightningRNNFilterModel(LightningModuleBase):
    """
    PytorchLightning wrapper for NODEFilterModel model.
    """
    def __init__(
        self,
        observable_dim: int,
        latent_dim: int,
        output_dim: Optional[int] = None,

        homogeneous: bool = False,
        bounded_variance: bool = False,
        rnn_update_layer: bool = False,

        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None,

        n_rnn_layers: int = 1,
        n_update_layers: int = 1,
        n_head_mlp_layers: int = 2,
        n_obs2latent_mlp_layers: int = 1,
        scale: float = 5.0,

        augmentation_remove_points_enable: bool = False,
        augmentation_remove_random_points_proba: float = 0.15,
        augmentation_remove_sequence_proba: float = 0.15,

        train_config: Optional[LightningTrainConfig] = None,
        log_epoch_metrics: bool = True
    ):
        if output_dim is None:
            output_dim = observable_dim

        super().__init__(train_config=train_config)

        self._model = RNNFilterModel(
            observable_dim=observable_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            homogeneous=homogeneous,
            bounded_variance=bounded_variance,
            rnn_update_layer=rnn_update_layer,
            n_rnn_layers=n_rnn_layers,
            n_update_layers=n_update_layers,
            n_head_mlp_layers=n_head_mlp_layers,
            n_obs2latent_mlp_layers=n_obs2latent_mlp_layers,
            scale=scale
        )

        self._augmentation_remove_points_enable = augmentation_remove_points_enable
        self._augmentation_remove_random_points_proba = augmentation_remove_random_points_proba
        self._augmentation_remove_sequence_proba = augmentation_remove_sequence_proba

        self._loss_func = factory_loss_function(train_config.loss_name, train_config.loss_params) \
            if train_config is not None else None
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
    def core(self) -> RNNFilterModel:
        return self._model

    def forward(self, x_obs: torch.Tensor, t_obs: torch.Tensor, x_unobs: torch.Tensor, t_unobs: torch.Tensor,
                mask: Optional[List[bool]] = None, *args, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._model(x_obs, t_obs, x_unobs, t_unobs, mask=mask, *args, **kwargs)

    def inference(self, x_obs: torch.Tensor, t_obs: torch.Tensor, x_unobs: torch.Tensor, t_unobs: torch.Tensor, mask_unobserved: bool = False, *args, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        mask = [mask_unobserved for _ in range(t_unobs.shape[0])]
        return self._model(x_obs, t_obs, x_unobs, t_unobs, mask=mask, *args, **kwargs)

    def _calc_loss_and_metrics(
        self,
        orig_bboxes_obs: torch.Tensor,
        orig_bboxes_unobs_prior: torch.Tensor,
        orig_bboxes_unobs_posterior: torch.Tensor,
        transformed_bboxes_unobs_prior: torch.Tensor,
        transformed_bboxes_unobs_posterior: torch.Tensor,
        bboxes_unobs_prior_mean: torch.Tensor,
        bboxes_unobs_prior_log_var: torch.Tensor,
        bboxes_unobs_posterior_mean: torch.Tensor,
        bboxes_unobs_posterior_log_var: torch.Tensor,
        metadata: dict
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Optional[Dict[str, float]]]:
        bboxes_unobs_prior_var = self._model.postprocess_log_var(bboxes_unobs_prior_log_var)
        bboxes_unobs_posterior_var = self._model.postprocess_log_var(bboxes_unobs_posterior_log_var)

        prior_loss = self._loss_func(bboxes_unobs_prior_mean, transformed_bboxes_unobs_prior, bboxes_unobs_prior_var)
        posterior_loss = self._loss_func(bboxes_unobs_posterior_mean, transformed_bboxes_unobs_posterior, bboxes_unobs_posterior_var)
        loss = {
            'prior_loss': prior_loss,
            'posterior_loss': posterior_loss,
            'loss': prior_loss + posterior_loss
        }

        if self._transform_func is not None:
            # Invert mean
            _, bboxes_unobs_prior_mean, *_ = self._transform_func.inverse([orig_bboxes_obs, bboxes_unobs_prior_mean, metadata, None], shallow=False)
            _, bboxes_unobs_posterior_mean, *_ = self._transform_func.inverse([orig_bboxes_obs, bboxes_unobs_posterior_mean, metadata, None],
                                                                              shallow=False)

        if not self._log_epoch_metrics:
            return loss, None

        gt_traj = orig_bboxes_unobs_prior.detach().cpu().numpy()
        prior_traj = bboxes_unobs_prior_mean.detach().cpu().numpy()
        prior_metrics = metrics_func(gt_traj, prior_traj)
        prior_metrics = {f'prior_{name}': value for name, value in prior_metrics.items()}

        gt_traj = orig_bboxes_unobs_posterior.detach().cpu().numpy()
        posterior_traj = bboxes_unobs_posterior_mean.detach().cpu().numpy()
        posterior_metrics = metrics_func(gt_traj, posterior_traj) if self._log_epoch_metrics else None
        posterior_metrics = {f'posterior_{name}': value for name, value in posterior_metrics.items()}
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

    def training_step(self, batch: Dict[str, Union[dict, torch.Tensor]], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_aug_unobs, ts_obs, ts_unobs, orig_bboxes_obs, orig_bboxes_unobs, bboxes_unobs, metadata = batch.values()

        bboxes_unobs_posterior = bboxes_unobs
        orig_bboxes_unobs_posterior = orig_bboxes_unobs
        mask: Optional[List[bool]] = None
        if self._augmentation_remove_points_enable:
            with torch.no_grad():
                # TODO: Move to augmentations
                # TODO: Write tests
                assert bboxes_aug_unobs.shape[0] >= 3, 'Minimum length of trajectory is 3 in order to perform special augmentations!'

                r = random.random()
                if r < self._augmentation_remove_random_points_proba:
                    mask = [True for _ in range(ts_unobs.shape[0])]
                    (bboxes_aug_unobs,), removed_indices, kept_indices = remove_points([bboxes_aug_unobs], min_length=1)
                    for i in removed_indices:
                        mask[i] = False
                    bboxes_unobs_posterior = bboxes_unobs_posterior[kept_indices, :, :]
                    orig_bboxes_unobs_posterior = orig_bboxes_unobs_posterior[kept_indices, :, :]

                elif r < self._augmentation_remove_random_points_proba + self._augmentation_remove_sequence_proba:
                    # TODO: Move to function
                    # TODO: Write tests
                    # Remove random sequence (connected points)
                    mask = [True for _ in range(ts_unobs.shape[0])]
                    n = bboxes_aug_unobs.shape[0]
                    start = random.randint(1, n - 1)  # s ~ U[1, n-1]
                    end = random.randint(start, n)  # e ~ U[s, n]
                    bboxes_aug_unobs = torch.cat([bboxes_aug_unobs[:start, :, :], bboxes_aug_unobs[end:n, :, :]], dim=0)
                    for i in range(start, end):
                        mask[i] = False
                    bboxes_unobs_posterior = torch.cat([bboxes_unobs_posterior[:start, :, :], bboxes_unobs_posterior[end:n, :, :]], dim=0)
                    orig_bboxes_unobs_posterior = \
                        torch.cat([orig_bboxes_unobs_posterior[:start, :, :], orig_bboxes_unobs_posterior[end:n, :, :]], dim=0)

        bboxes_prior_mean, bboxes_prior_log_var, bboxes_posterior_mean, bboxes_posterior_log_var, _ = \
            self.forward(bboxes_obs, ts_obs, bboxes_aug_unobs, ts_unobs, mask=mask)

        # noinspection PyTypeChecker
        loss, metrics = self._calc_loss_and_metrics(
            orig_bboxes_obs=orig_bboxes_obs,
            orig_bboxes_unobs_prior=orig_bboxes_unobs,
            orig_bboxes_unobs_posterior=orig_bboxes_unobs_posterior,
            transformed_bboxes_unobs_prior=bboxes_unobs,
            transformed_bboxes_unobs_posterior=bboxes_unobs_posterior,
            bboxes_unobs_prior_mean=bboxes_prior_mean,
            bboxes_unobs_prior_log_var=bboxes_prior_log_var,
            bboxes_unobs_posterior_mean=bboxes_posterior_mean,
            bboxes_unobs_posterior_log_var=bboxes_posterior_log_var,
            metadata=metadata
        )
        self._log_loss(loss, prefix='training', log_step=True)
        self._log_metrics(metrics, prefix='training')

        return loss

    def validation_step(self, batch: Dict[str, Union[dict, torch.Tensor]], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_aug_unobs, ts_obs, ts_unobs, orig_bboxes_obs, orig_bboxes_unobs, bboxes_unobs, metadata = batch.values()

        bboxes_prior_mean, bboxes_prior_log_var, bboxes_posterior_mean, bboxes_posterior_log_var, _ = \
            self.forward(bboxes_obs, ts_obs, bboxes_aug_unobs, ts_unobs)

        # noinspection PyTypeChecker
        loss, metrics = self._calc_loss_and_metrics(
            orig_bboxes_obs=orig_bboxes_obs,
            orig_bboxes_unobs_prior=orig_bboxes_unobs,
            orig_bboxes_unobs_posterior=orig_bboxes_unobs,
            transformed_bboxes_unobs_prior=bboxes_unobs,
            transformed_bboxes_unobs_posterior=bboxes_unobs,
            bboxes_unobs_prior_mean=bboxes_prior_mean,
            bboxes_unobs_prior_log_var=bboxes_prior_log_var,
            bboxes_unobs_posterior_mean=bboxes_posterior_mean,
            bboxes_unobs_posterior_log_var=bboxes_posterior_log_var,
            metadata=metadata
        )
        self._log_loss(loss, prefix='val', log_step=False)
        self._log_metrics(metrics, prefix='val')

        return loss

def run_test() -> None:
    rfm = RNNFilterModel(
        observable_dim=4,
        latent_dim=3,
        output_dim=4,

        n_rnn_layers=2
    )

    x_obs = torch.randn(4, 3, 4)
    x_unobs = torch.randn(2, 3, 4)
    t_obs = torch.tensor([0, 1, 2, 4], dtype=torch.float32).view(-1, 1, 1).repeat(1, 3, 1)
    t_unobs = torch.tensor([6, 9], dtype=torch.float32).view(-1, 1, 1).repeat(1, 3, 1)

    priors_mean, priors_log_var, posteriors_mean, posteriors_log_var = rfm(x_obs, t_obs, x_unobs, t_unobs)
    print(priors_mean.shape, priors_log_var.shape, posteriors_mean.shape, posteriors_log_var.shape)


if __name__ == '__main__':
    run_test()
