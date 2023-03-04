"""
BBox transformations. Contains
- BBox trajectory first order difference transformation
- BBox coordination standardization to N(0, 1)
- BBox trajectory standardized first order difference transformation
"""

import torch
from typing import Union, List

from nodetracker.datasets.transforms.base import InvertibleTransform, TensorCollection


class BboxFirstOrderDifferenceTransform(InvertibleTransform):
    """
    Applies first difference transformation:
    Y[i] = X[i] - X[i-1]
    Y[0] is removed
    """
    def __init__(self):
        super().__init__(name='first_difference')

    def apply(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        bbox_obs, bbox_unobs, ts_obs, *other = data
        assert bbox_obs.shape[0] >= 0, f'{self.__name__} requires at least 2 observable points. ' \
                                       f'Found {bbox_obs.shape[0]}'
        if not shallow:
            bbox_obs = bbox_obs.clone()
            bbox_unobs = bbox_unobs.clone() if bbox_unobs is not None else None
            ts_obs = ts_obs.clone()

        if bbox_unobs is not None:
            # During live inference, bbox_unobs are not known (this is for training only)
            bbox_unobs[1:, ...] = bbox_unobs[1:, ...] - bbox_unobs[:-1, ...]
            bbox_unobs[0, ...] = bbox_unobs[0, ...] - bbox_obs[-1, ...]

        bbox_obs[1:, ...] = bbox_obs[1:, ...] - bbox_obs[:-1, ...]
        bbox_obs, ts_obs = bbox_obs[1:, ...], ts_obs[1:, ...]  # Dump first

        return bbox_obs, bbox_unobs, ts_obs, *other

    def inverse(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        orig_bbox_obs, bbox_hat, *other = data
        if not shallow:
            bbox_hat = bbox_hat.clone()

        bbox_hat[0, ...] = bbox_hat[0, ...] + orig_bbox_obs[-1, ...]
        bbox_hat = torch.cumsum(bbox_hat, dim=0)

        return orig_bbox_obs, bbox_hat, *other


class BBoxStandardizationTransform(InvertibleTransform):
    """
    Applies standardization transformation:
    Y[i] = (X[i] - mean(X)) / std(X)
    """
    def __init__(self, mean: Union[float, List[float]], std: Union[float, List[float]]):
        super().__init__(name='standardization')
        self._mean = mean
        self._std = std

        if isinstance(self._mean, float):
            # Convert to list
            self._mean = [self._mean] * 4

        if isinstance(self._std, float):
            # Convert to list
            self._std = [self._std] * 4

        self._mean = torch.tensor(self._mean, dtype=torch.float32)
        self._std = torch.tensor(self._std, dtype=torch.float32)

    def apply(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        bbox_obs, bbox_unobs, *other = data

        bbox_obs = (bbox_obs - self._mean) / self._std
        # During live inference, bbox_unobs are not known (this is for training only)
        bbox_unobs = (bbox_unobs - self._mean) / self._std if bbox_unobs is not None else None

        return bbox_obs, bbox_unobs, *other

    def inverse(self, data: TensorCollection, shallow: bool = True, n_samples: int = 1) -> TensorCollection:
        bbox_obs, bbox_unobs, *other = data

        if n_samples == 1:
            # Note: inverse transform is not applied to `bbox_obs`
            bbox_unobs = bbox_unobs * self._std + self._mean
        else:
            # Support (Improvisation) for VAE monte carlo sampling for mean and std estimation
            mean_repeated = self._mean.repeat(n_samples)
            std_repeated = self._std.repeat(n_samples)
            bbox_unobs = bbox_unobs * std_repeated + mean_repeated

        return [bbox_obs, bbox_unobs, *other]


class BBoxStandardizedFirstOrderDifferenceTransform(InvertibleTransform):
    """
    Step 1: Applies first difference transformation:
    Y[i] = X[i] - X[i-1]
    Y[0] is removed

    Step 2: Applies standardization transformation:
    Z[i] = (Y[i] - mean(Y)) / std(Y)
    """
    def __init__(self, mean: float, std: float):
        super().__init__(name='standardized_first_difference')
        self._first_difference = BboxFirstOrderDifferenceTransform()
        self._standardization = BBoxStandardizationTransform(mean=mean, std=std)

    def apply(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        data = self._first_difference.apply(data, shallow=shallow)
        data = self._standardization.apply(data, shallow=False)
        return data

    def inverse(self, data: TensorCollection, shallow: bool = True, n_samples: int = 1) -> TensorCollection:
        data = self._standardization.inverse(data, shallow=shallow, n_samples=n_samples)
        data = self._first_difference.inverse(data, shallow=False)
        return data


# noinspection DuplicatedCode
def run_test_first_difference() -> None:
    bbox_obs = torch.randn(2, 2, 4)
    bbox_unobs = torch.randn(3, 2, 4)
    ts_obs = torch.randn(2, 2, 1)
    ts_unobs = torch.randn(3, 2, 1)
    first_diff = BboxFirstOrderDifferenceTransform()

    transformed_bbox_obs, transformed_bbox_unobs, _, _ = \
        first_diff.apply([bbox_obs, bbox_unobs, ts_obs, ts_unobs], shallow=False)
    assert transformed_bbox_obs.shape == (1, 2, 4)
    assert transformed_bbox_unobs.shape == (3, 2, 4)

    _, inv_transformed_bbox_unobs = first_diff.inverse([bbox_obs, transformed_bbox_unobs])
    assert torch.abs(inv_transformed_bbox_unobs - bbox_unobs).sum().item() < 1e-3


def run_standardization() -> None:
    bbox_obs = torch.randn(2, 2, 4)
    bbox_unobs = torch.randn(3, 2, 4)
    ts_obs = torch.randn(2, 2, 1)
    ts_unobs = torch.randn(3, 2, 1)
    first_diff = BBoxStandardizationTransform(1, 2.0)

    _, transformed_bbox_unobs, _, _ = \
        first_diff.apply([bbox_obs, bbox_unobs, ts_obs, ts_unobs], shallow=False)

    _, inv_transformed_bbox_unobs = first_diff.inverse([bbox_obs, transformed_bbox_unobs])
    assert torch.abs(inv_transformed_bbox_unobs - bbox_unobs).sum().item() < 1e-3


# noinspection DuplicatedCode
def run_test_standardized_first_difference() -> None:
    bbox_obs = torch.randn(2, 2, 4)
    bbox_unobs = torch.randn(3, 2, 4)
    ts_obs = torch.randn(2, 2, 1)
    ts_unobs = torch.randn(3, 2, 1)
    first_diff = BBoxStandardizedFirstOrderDifferenceTransform(1, 2.0)

    transformed_bbox_obs, transformed_bbox_unobs, _, _ = \
        first_diff.apply([bbox_obs, bbox_unobs, ts_obs, ts_unobs], shallow=False)
    assert transformed_bbox_obs.shape == (1, 2, 4)
    assert transformed_bbox_unobs.shape == (3, 2, 4)

    _, inv_transformed_bbox_unobs = first_diff.inverse([bbox_obs, transformed_bbox_unobs])
    assert torch.abs(inv_transformed_bbox_unobs - bbox_unobs).sum().item() < 1e-3


if __name__ == '__main__':
    run_test_first_difference()
    run_standardization()
    run_test_standardized_first_difference()
