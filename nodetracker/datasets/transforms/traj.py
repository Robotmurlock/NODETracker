from typing import Collection

import torch

from nodetracker.datasets.transforms.base import InvertibleTransform


class BboxFirstDifferenceTransform(InvertibleTransform):
    """
    Applies first difference transformation:
    Y[i] = X[i] - X[i-1]
    Y[0] is removed
    """
    def __init__(self):
        super().__init__(name='first_difference')

    def apply(self, data: Collection[torch.Tensor], shallow: bool = True) -> Collection[torch.Tensor]:
        bbox_obs, bbox_unobs, ts_obs, *other = data
        assert bbox_obs.shape[0] >= 0, f'{self.__name__} requires at least 2 observable points. Found {bbox_obs.shape[0]}'
        if not shallow:
            bbox_obs = bbox_obs.clone()
            bbox_unobs = bbox_unobs.clone()
            ts_obs = ts_obs.clone()

        bbox_unobs[1:, ...] = bbox_unobs[1:, ...] - bbox_unobs[:-1, ...]
        bbox_unobs[0, ...] = bbox_unobs[0, ...] - bbox_obs[-1, ...]
        bbox_obs[1:, ...] = bbox_obs[1:, ...] - bbox_obs[:-1, ...]
        bbox_obs, ts_obs = bbox_obs[1:, ...], ts_obs[1:, ...]  # Dump first

        return bbox_obs, bbox_unobs, ts_obs, *other

    def inverse(self, data: Collection[torch.Tensor], shallow: bool = True) -> Collection[torch.Tensor]:
        orig_bbox_obs, bbox_hat, *other = data
        if not shallow:
            bbox_hat = bbox_hat.clone()

        bbox_hat[0, ...] = bbox_hat[0, ...] + orig_bbox_obs[-1, ...]
        bbox_hat = torch.cumsum(bbox_hat, dim=0)
        return orig_bbox_obs, bbox_hat, *other


def run_test() -> None:
    bbox_obs = torch.randn(2, 2, 4)
    bbox_unobs = torch.randn(3, 2, 4)
    ts_obs = torch.randn(2, 2, 1)
    ts_unobs = torch.randn(3, 2, 1)
    first_diff = BboxFirstDifferenceTransform()

    transformed_bbox_obs, transformed_bbox_unobs, transformed_ts_obs, transformed_ts_unobs = \
        first_diff.apply([bbox_obs, bbox_unobs, ts_obs, ts_unobs], shallow=False)
    assert transformed_bbox_obs.shape == (1, 2, 4)
    assert transformed_bbox_unobs.shape == (3, 2, 4)

    _, inv_transformed_bbox_unobs = first_diff.inverse([bbox_obs, transformed_bbox_unobs])
    assert torch.abs(inv_transformed_bbox_unobs - bbox_unobs).sum().item() < 1e-3


if __name__ == '__main__':
    run_test()

