"""
Implementation of a custom Conditional Neural Processes with reccurent aggregation
"""
from typing import Optional, Union, Tuple, Dict

import torch
from torch import nn

from nodetracker.datasets.transforms import InvertibleTransform, InvertibleTransformWithVariance
from nodetracker.evaluation.metrics.sot import metrics_func
from nodetracker.node.losses.factory import factory_loss_function
from nodetracker.node.odernn.utils import LightningGaussianModel, run_simple_lightning_guassian_model_test
from nodetracker.node.utils.training import extract_mean_and_var
from nodetracker.node.utils import LightningTrainConfig, LightningModuleBase
from nodetracker.np.core.rnn_cnp import RNNCNP, RNNCNPFilter
from nodetracker.np.utils import to_scaled_relative_ts


class BaselineRNNCNP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,

        hidden_dim: int = 8,
        n_input2hidden_layers: int = 2,
        n_target2hidden_layers: int = 2,
        n_enc_layers: int = 2,

        n_head_layers: int = 2,
        n_agg_layers: int = 1,
        t_scale: float = 5.0
    ):
        super().__init__()

        self._rnn_cnp = RNNCNP(
            input_dim=input_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            n_input2hidden_layers=n_input2hidden_layers,
            n_target2hidden_layers=n_target2hidden_layers,
            n_enc_layers=n_enc_layers,
            n_head_layers=n_head_layers,
            n_agg_layers=n_agg_layers
        )

        self._t_scale = t_scale

    def forward(self, x_obs: torch.Tensor, ts_obs: torch.Tensor, ts_unobs: torch.Tensor, metadata: dict) -> torch.Tensor:
        with torch.no_grad():
            ts_obs, ts_unobs = to_scaled_relative_ts(ts_obs, ts_unobs, self._t_scale)

        _ = metadata  # Ignored
        return self._rnn_cnp(ts_obs, x_obs, ts_unobs)


class LightningBaselineRNNCNP(LightningGaussianModel):
    """
    Trainer wrapper for RNNCNP.
    """
    def __init__(
        self,
        observable_dim: int,
        hidden_dim: int,

        n_input2hidden_layers: int = 2,
        n_target2hidden_layers: int = 2,
        n_enc_layers: int = 2,

        n_head_layers: int = 2,
        n_agg_layers: int = 1,

        t_scale: float = 5.0,

        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None,

        train_config: Optional[LightningTrainConfig] = None,
        log_epoch_metrics: bool = True
    ):
        """
        Args:
            observable_dim: Trajectory point dimension
            hidden_dim: Hidden trajectory dimension
        """
        model = BaselineRNNCNP(
            input_dim=1,
            target_dim=observable_dim,
            hidden_dim=hidden_dim,
            n_input2hidden_layers=n_input2hidden_layers,
            n_target2hidden_layers=n_target2hidden_layers,
            n_enc_layers=n_enc_layers,
            n_head_layers=n_head_layers,
            n_agg_layers=n_agg_layers,
            t_scale=t_scale
        )
        super().__init__(
            train_config=train_config,
            model=model,
            model_gaussian=True,
            transform_func=transform_func,
            log_epoch_metrics=log_epoch_metrics
        )

class LightningRNNCNPFilter(LightningModuleBase):
    """
    PytorchLightning wrapper for RNNCNPFilter model.
    """
    def __init__(
        self,
        observable_dim: int,
        hidden_dim: int,

        n_input2hidden_layers: int = 2,
        n_target2hidden_layers: int = 2,
        n_enc_layers: int = 2,

        n_head_layers: int = 2,
        n_agg_layers: int = 2,
        t_scale: float = 5.0,
        bounded_variance: bool = False,
        bounded_value: float = 0.1,

        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None,

        train_config: Optional[LightningTrainConfig] = None,
        log_epoch_metrics: bool = True
    ):
        super().__init__(train_config=train_config)
        self._model = RNNCNPFilter(
            input_dim=1,
            target_dim=observable_dim,
            hidden_dim=hidden_dim,
            n_input2hidden_layers=n_input2hidden_layers,
            n_target2hidden_layers=n_target2hidden_layers,
            n_enc_layers=n_enc_layers,
            n_head_layers=n_head_layers,
            n_agg_layers=n_agg_layers,
            t_scale=t_scale,
            bounded_variance=bounded_variance,
            bounded_value=bounded_value
        )

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
    def core(self) -> RNNCNPFilter:
        return self._model

    def forward(self, x_obs: torch.Tensor, t_obs: torch.Tensor, x_unobs: torch.Tensor, t_unobs: torch.Tensor,
                *args, **kwargs) \
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

    def _calc_loss_and_metrics(
        self,
        orig_bboxes_obs: torch.Tensor,
        orig_bboxes_unobs_prior: torch.Tensor,
        orig_bboxes_unobs_posterior: torch.Tensor,
        transformed_bboxes_unobs_prior: torch.Tensor,
        transformed_bboxes_unobs_posterior: torch.Tensor,
        bboxes_unobs_prior_mean: torch.Tensor,
        bboxes_unobs_prior_var: torch.Tensor,
        bboxes_unobs_posterior_mean: torch.Tensor,
        bboxes_unobs_posterior_var: torch.Tensor,
        metadata: dict
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Optional[Dict[str, float]]]:
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
        bboxes_prior, bboxes_posterior = self.forward(bboxes_obs, ts_obs, bboxes_aug_unobs, ts_unobs, metadata)
        bboxes_prior_mean, bboxes_prior_var = self._model.unpack_output(bboxes_prior)
        bboxes_posterior_mean, bboxes_posterior_var = self._model.unpack_output(bboxes_posterior)

        # noinspection PyTypeChecker
        loss, metrics = self._calc_loss_and_metrics(
            orig_bboxes_obs=orig_bboxes_obs,
            orig_bboxes_unobs_prior=orig_bboxes_unobs,
            orig_bboxes_unobs_posterior=orig_bboxes_unobs,
            transformed_bboxes_unobs_prior=bboxes_unobs,
            transformed_bboxes_unobs_posterior=bboxes_unobs,
            bboxes_unobs_prior_mean=bboxes_prior_mean,
            bboxes_unobs_prior_var=bboxes_prior_var,
            bboxes_unobs_posterior_mean=bboxes_posterior_mean,
            bboxes_unobs_posterior_var=bboxes_posterior_var,
            metadata=metadata
        )
        self._log_loss(loss, prefix='training', log_step=True)
        self._log_metrics(metrics, prefix='training')

        return loss

    def validation_step(self, batch: Dict[str, Union[dict, torch.Tensor]], *args, **kwargs) -> torch.Tensor:
        bboxes_obs, bboxes_aug_unobs, ts_obs, ts_unobs, orig_bboxes_obs, orig_bboxes_unobs, bboxes_unobs, metadata = batch.values()
        bboxes_prior, bboxes_posterior = self.forward(bboxes_obs, ts_obs, bboxes_aug_unobs, ts_unobs, metadata)
        bboxes_prior_mean, bboxes_prior_var = extract_mean_and_var(bboxes_prior)
        bboxes_posterior_mean, bboxes_posterior_var = extract_mean_and_var(bboxes_posterior)

        # noinspection PyTypeChecker
        loss, metrics = self._calc_loss_and_metrics(
            orig_bboxes_obs=orig_bboxes_obs,
            orig_bboxes_unobs_prior=orig_bboxes_unobs,
            orig_bboxes_unobs_posterior=orig_bboxes_unobs,
            transformed_bboxes_unobs_prior=bboxes_unobs,
            transformed_bboxes_unobs_posterior=bboxes_unobs,
            bboxes_unobs_prior_mean=bboxes_prior_mean,
            bboxes_unobs_prior_var=bboxes_prior_var,
            bboxes_unobs_posterior_mean=bboxes_posterior_mean,
            bboxes_unobs_posterior_var=bboxes_posterior_var,
            metadata=metadata
        )
        self._log_loss(loss, prefix='val', log_step=False)
        self._log_metrics(metrics, prefix='val')

        return loss


def main() -> None:
    run_simple_lightning_guassian_model_test(
        model_class=LightningBaselineRNNCNP,
        params={
            'observable_dim': 7,
            'hidden_dim': 3,
        },
        model_gaussian_only=True
    )


if __name__ == '__main__':
    main()
