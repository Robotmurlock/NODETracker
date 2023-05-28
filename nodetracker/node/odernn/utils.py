"""
Support for Gaussian models.
"""
from typing import Optional, Tuple, Type, Dict, Any, Union

import torch
from torch import nn

from nodetracker.datasets.transforms import InvertibleTransform, InvertibleTransformWithVariance
from nodetracker.node.utils.training import LightningModuleForecaster, LightningTrainConfig


class LightningGaussianModel(LightningModuleForecaster):
    """
    PytorchLightning wrapper for recurrent like guassian model.
    """
    def __init__(
        self,
        model: nn.Module,
        model_gaussian: bool = False,
        train_config: Optional[LightningTrainConfig] = None,
        transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None
    ):
        super().__init__(
            train_config=train_config,
            model=model,
            model_gaussian=model_gaussian,
            transform_func=transform_func
        )

        self._is_modeling_gaussian = model_gaussian

    @property
    def is_modeling_gaussian(self) -> bool:
        """
        Checks if model estimates Gaussian or not.

        Returns:
            True if model estimates Gaussian
            else False (model just predicts values)
        """
        return self._is_modeling_gaussian

    @staticmethod
    def extract_mean_and_std(bboxes_unobs_hat: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function for Gaussian model postprocess

        Args:
            bboxes_unobs_hat: Prediction

        Returns:
            bboxes_hat_mean, bboxes_hat_std
        """
        bboxes_unobs_hat = bboxes_unobs_hat.view(*bboxes_unobs_hat.shape[:-1], -1, 2)
        bboxes_unobs_hat_mean = bboxes_unobs_hat[..., 0]
        bboxes_unobs_hat_log_var = bboxes_unobs_hat[..., 1]
        bboxes_unobs_hat_var = torch.exp(bboxes_unobs_hat_log_var)
        bboxes_unobs_hat_std = torch.sqrt(bboxes_unobs_hat_var)

        return bboxes_unobs_hat_mean, bboxes_unobs_hat_std

    def inference(self, x: torch.Tensor, t_obs: torch.Tensor, t_unobs: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, ...]:
        output = self._model(x, t_obs, t_unobs)
        if isinstance(output, tuple):
            # Case 1: Model outputs single value
            x_hat, *other = output
        else:
            # Case 2: Model outputs a tuple
            x_hat = output
            other = tuple()

        if self._model_gaussian:
            x_hat_mean, x_hat_std = self.extract_mean_and_std(x_hat)
            return x_hat_mean, x_hat_std, *other

        return x_hat, *other


def run_simple_lightning_guassian_model_test(model_class: Type[LightningGaussianModel], params: Dict[str, Any]):
    """
    Performs tests on implemented inherited LightningGaussianModel model.

    Args:
        model_class: Model class
        params: Parameters
    """
    # Test Model
    xs = torch.randn(4, 3, 7)
    ts_obs = torch.randn(4, 3, 1)
    ts_unobs = torch.randn(2, 3, 1)
    expected_shape = (2, 3, 7)

    # Test standard (no gaussian)
    model = model_class(**params)

    output = model(xs, ts_obs, ts_unobs)
    if isinstance(output, tuple):
        output, *_ = output
    assert output.shape == expected_shape, f'Expected shape {expected_shape} but found {output.shape}!'

    # Test model (gaussian)
    model = model_class(model_gaussian=True, **params)

    expected_shape = (2, 3, 14)
    output = model(xs, ts_obs, ts_unobs)
    if isinstance(output, tuple):
        output, *_ = output
    assert output.shape == expected_shape, f'Expected shape {expected_shape} but found {output.shape}!'

    expected_shapes = [(2, 3, 7), (2, 3, 7)]
    output = model.extract_mean_and_std(output)
    output_shapes = [o.shape for o in output]
    assert output_shapes == expected_shapes, f'Expected shape {expected_shape} but found {output_shapes}!'
