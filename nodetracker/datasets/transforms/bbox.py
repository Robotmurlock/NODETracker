"""
BBox transformations. Contains
- BBox trajectory first order difference transformation
- BBox coordination standardization to N(0, 1)
- BBox trajectory standardized first order difference transformation
- BBox trajectory relative to the last observed point
- BBox trajectory standardized relative to the last observed point transformation
- BBox transform composition
- BBox coordination standardization to N(0, 1) (by category)
"""

from typing import Union, List, Optional, Dict, Tuple
import copy

import torch

from nodetracker.datasets.transforms.base import InvertibleTransformWithVariance, TensorCollection
from nodetracker.utils.lookup import LookupTable


class BboxFirstOrderDifferenceTransform(InvertibleTransformWithVariance):
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

        assert not shallow
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

    def inverse_std(
        self,
        t_std: torch.Tensor,
        additional_data: Optional[TensorCollection] = None,
        shallow: bool = True
    ) -> TensorCollection:
        # Check `inverse_var`
        t_var = torch.square(t_std)
        t_var_cumsum = torch.cumsum(t_var, dim=0)
        t_std_cumsum = torch.sqrt(t_var_cumsum)
        return t_std_cumsum

    def inverse_var(
        self,
        t_var: torch.Tensor,
        additional_data: Optional[TensorCollection] = None,
        shallow: bool = True
    ) -> TensorCollection:
        # In order to estimate std for inverse two assumptions are added
        # 1. Random variables y[i-1] and y[i] are independent
        # 2. Variance of last observed bbox coordinate is 0 (i.e. var(x[-1]) = 0)
        # Let: y - transformed, x - original, 0 - first unobserved time point
        # Transformation: y[i] = x[i] - x[i-1]
        # Inverse transformation: x[i] = y[i] + x[i-1]
        # Known variances: y[i] and x[-1]
        # => var(x[0]) = var(y[0]) + var(x[-1]) = var(y[0])  # from (1) and (2)
        # => var(x[1]) = var(y[1]) + var(x[0]) = var(y[0]) + var(y[1]))  # from (1)
        # ...
        # => var(x[i]) = var(y[i]) + var(x[i-1]) = sum[j=0,i] var(y[j])
        return torch.cumsum(t_var, dim=0)


class BBoxStandardizationTransform(InvertibleTransformWithVariance):
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
        mean = self._mean.to(bbox_obs)
        std = self._std.to(bbox_obs)

        bbox_obs = (bbox_obs - mean) / std
        # During live inference, bbox_unobs are not known (this is for training only)
        bbox_unobs = (bbox_unobs - mean) / std if bbox_unobs is not None else None

        return bbox_obs, bbox_unobs, *other

    def inverse(self, data: TensorCollection, shallow: bool = True, n_samples: int = 1) -> TensorCollection:
        bbox_obs, bbox_unobs, *other = data
        mean = self._mean.to(bbox_obs)
        std = self._std.to(bbox_obs)

        if n_samples == 1:
            # Note: inverse transform is not applied to `bbox_obs`
            bbox_unobs = bbox_unobs * std + mean
        else:
            # Support (Improvisation) for VAE monte carlo sampling for mean and std estimation
            mean_repeated = mean.repeat(n_samples)
            std_repeated = std.repeat(n_samples)
            bbox_unobs = bbox_unobs * std_repeated + mean_repeated

        return bbox_obs, bbox_unobs, *other

    def inverse_std(
        self,
        t_std: torch.Tensor,
        additional_data: Optional[TensorCollection] = None,
        shallow: bool = True
    ) -> TensorCollection:
        # Std of inverse transformation in this case is trivial to calculate
        # y - transformed, x - original
        # y[i] = (x[i] - m) / s
        # => x[i] = y[i] * s + m
        # => std(x) = std(y) * s
        std = self._std.to(t_std)
        return t_std * std

    def inverse_var(
        self,
        t_var: torch.Tensor,
        additional_data: Optional[TensorCollection] = None,
        shallow: bool = True
    ) -> TensorCollection:
        # Similar to `inverse_std`
        std = self._std.to(t_var)
        return t_var * torch.square(std)


class BBoxCompositeTransform(InvertibleTransformWithVariance):
    def __init__(self, transforms: List[InvertibleTransformWithVariance]):
        super().__init__(name='composite')
        self._transforms = transforms

    def apply(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        for t in self._transforms:
            data = t.apply(data, shallow=shallow)
        return data

    def inverse(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        for t in self._transforms[::-1]:
            data = t.inverse(data, shallow=shallow)
        return data

    def inverse_std(
        self,
        t_std: torch.Tensor,
        additional_data: Optional[TensorCollection] = None,
        shallow: bool = True
    ) -> TensorCollection:
        for t in self._transforms[::-1]:
            t_std = t.inverse_std(t_std, additional_data=additional_data, shallow=shallow)
        return t_std

    def inverse_var(
        self,
        t_var: torch.Tensor,
        additional_data: Optional[TensorCollection] = None,
        shallow: bool = True
    ) -> TensorCollection:
        for t in self._transforms[::-1]:
            t_var = t.inverse_var(t_var, additional_data=additional_data, shallow=shallow)
        return t_var


class BBoxStandardizedFirstOrderDifferenceTransform(BBoxCompositeTransform):
    """
    This is still here for back-compatibility...

    Step 1: Applies first difference transformation:
    Y[i] = X[i] - X[i-1]
    Y[0] is removed

    Step 2: Applies standardization transformation:
    Z[i] = (Y[i] - mean(Y)) / std(Y)
    """
    def __init__(self, mean: Union[float, List[float]], std: Union[float, List[float]]):
        transforms = [
            BboxFirstOrderDifferenceTransform(),
            BBoxStandardizationTransform(mean=mean, std=std)
        ]
        super().__init__(transforms=transforms)


class BBoxRelativeToLastObsTransform(InvertibleTransformWithVariance):
    """
    * For all observed coordinates:
    Y[i] = X[-1] - X[i]
    Y[-1] is removed

    * For all unobserved coordinates:
    Y[i] = X[i] - X[-1]
    """
    def __init__(self):
        super().__init__(name='relative_to_last_obs')

    def apply(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        bbox_obs, bbox_unobs, ts_obs, *other = data
        last_obs = bbox_obs[-1:]

        if not shallow:
            bbox_obs = bbox_obs.clone()
            bbox_unobs = bbox_unobs.clone() if bbox_unobs is not None else None

        bbox_obs, ts_obs = bbox_obs[:-1], ts_obs[:-1]  # Last element becomes redundant
        bbox_obs = last_obs.expand_as(bbox_obs) - bbox_obs
        bbox_unobs = bbox_unobs - last_obs.expand_as(bbox_unobs) if bbox_unobs is not None else None

        return bbox_obs, bbox_unobs, ts_obs, *other

    def inverse(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        orig_bbox_obs, bbox_hat, *other = data
        last_obs = orig_bbox_obs[-1:]

        if not shallow:
            bbox_hat = bbox_hat.clone()

        bbox_hat = bbox_hat + last_obs.expand_as(bbox_hat)

        return orig_bbox_obs, bbox_hat, *other

    def inverse_std(
        self,
        t_std: torch.Tensor,
        additional_data: Optional[TensorCollection] = None,
        shallow: bool = True
    ) -> TensorCollection:
        return t_std  # It's assumed that last observed element has variance equal to 0

    def inverse_var(
        self,
        t_var: torch.Tensor,
        additional_data: Optional[TensorCollection] = None,
        shallow: bool = True
    ) -> TensorCollection:
        return t_var  # It's assumed that last observed element has variance equal to 0


class BBoxStandardizedRelativeToLastObsTransform(BBoxCompositeTransform):
    """
    This is still here for back-compatibility...

    Step 1: Applies first difference transformation:
    * For all observed coordinates:
    Y[i] = X[-1] - X[i]
    Y[-1] is removed

    * For all unobserved coordinates:
    Y[i] = X[i] - X[-1]

    Step 2: Applies standardization transformation:
    Z[i] = (Y[i] - mean(Y)) / std(Y)
    """
    def __init__(self, mean: float, std: float):
        transforms = [
            BBoxRelativeToLastObsTransform(),
            BBoxStandardizationTransform(mean=mean, std=std)
        ]

        super().__init__(transforms=transforms)


class BBoxAddLabelTransform(InvertibleTransformWithVariance):
    def __init__(
        self,
        token_to_index: Optional[dict] = None,
        unknown_token: str = '<unk>',
        add_unknown_token: bool = True
    ):
        super().__init__(name='bbox_add_one_hot_label')
        self._lookup = LookupTable(
            token_to_index=token_to_index,
            unknown_token=unknown_token,
            add_unknown_token=add_unknown_token
        )

    def apply(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        bboxes_obs, bboxes_unobs, frame_ts_obs, frame_ts_unobs, orig_bbox_obs, metadata, *other = data
        category = metadata['category']
        category_index = self._lookup[category] * torch.ones(*bboxes_obs.shape[:-1], 1, dtype=torch.float32)
        bboxes_obs = torch.concat([bboxes_obs, category_index], dim=-1)
        return bboxes_obs, bboxes_unobs, frame_ts_obs, frame_ts_unobs, metadata, *other

    def inverse(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        return data

    def inverse_std(
        self,
        t_std: torch.Tensor,
        additional_data: Optional[TensorCollection] = None,
        shallow: bool = True
    ) -> TensorCollection:
        return t_std

    def inverse_var(
        self,
        t_var: torch.Tensor,
        additional_data: Optional[TensorCollection] = None,
        shallow: bool = True
    ) -> TensorCollection:
        return t_var


class BBoxCategoryStandardizationTransform(InvertibleTransformWithVariance):
    """
    Applies standardization transformation by category:
    Y[i] = (X[i] - mean_category(X)) / std_category(X)
    """
    def __init__(self, stats: Dict[str, Dict[str, Union[float, List[float], torch.Tensor]]]):
        super().__init__(name='standardization_by_category')
        stats = copy.deepcopy(stats)

        for stat in stats.values():
            assert set(stat.keys()) == {'mean', 'std'}, \
                f'Expected "mean" and "std" keys but found {list(stats.keys())}!'

            if isinstance(stat['mean'], float):
                # Convert to list
                stat['mean'] = [stat['mean']] * 4

            if isinstance(stat['std'], float):
                # Convert to list
                stat['std'] = [stat['std']] * 4

            stat['mean'] = torch.tensor(stat['mean'], dtype=torch.float32)
            stat['std'] = torch.tensor(stat['std'], dtype=torch.float32)

        self._stats = stats

    def _get_mean_std(self, category: str, device: Union[torch.device, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches mean and std stats for queries category.

        Args:
            category: Category

        Returns:
            Mean, Std
        """
        mean = self._stats[category]['mean'].to(device)
        std = self._stats[category]['std'].to(device)
        return mean, std

    def apply(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        bbox_obs, bbox_unobs, ts_obs, ts_unobs, metadata, *other = data
        if len(bbox_obs.shape) == 3:
            # Batch operation
            n_batches = bbox_obs.shape[1]
            for batch_index in range(n_batches):
                mean, std = self._get_mean_std(metadata['category'][batch_index], bbox_obs.device)

                bbox_obs[:, batch_index, :] = (bbox_obs[:, batch_index, :] - mean) / std
                # During live inference, bbox_unobs are not known (this is for training only)
                bbox_unobs[:, batch_index, :] = (bbox_unobs[:, batch_index, :] - mean) / std \
                    if bbox_unobs is not None else None
        elif len(bbox_obs.shape) == 2:
            # Single sample operation
            mean, std = self._get_mean_std(metadata['category'], bbox_obs.device)

            bbox_obs = (bbox_obs - mean) / std
            # During live inference, bbox_unobs are not known (this is for training only)
            bbox_unobs = (bbox_unobs - mean) / std if bbox_unobs is not None else None
        else:
            raise AssertionError(f'Invalid bbox shape {bbox_obs.shape}')

        return bbox_obs, bbox_unobs, ts_obs, ts_unobs, metadata, *other

    def inverse(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        bbox_obs, bbox_unobs, metadata, *other = data

        if len(bbox_obs.shape) == 3:
            # Batch operation
            n_batches = bbox_obs.shape[1]
            for batch_index in range(n_batches):
                mean, std = self._get_mean_std(metadata['category'][batch_index], bbox_obs.device)

                bbox_unobs[:, batch_index, :] = bbox_unobs[:, batch_index, :] * std + mean
        elif len(bbox_obs.shape) == 2:
            # Single sample operation
            mean, std = self._get_mean_std(metadata['category'], bbox_obs.device)

            bbox_unobs = bbox_unobs * std + mean
        else:
            raise AssertionError(f'Invalid bbox shape {bbox_obs.shape}')

        return bbox_obs, bbox_unobs, *other

    def inverse_std(
        self,
        t_std: torch.Tensor,
        additional_data: Optional[TensorCollection] = None,
        shallow: bool = True
    ) -> TensorCollection:
        metadata, *other = additional_data
        new_std = None

        if len(t_std.shape) == 3:
            # Batch operation
            n_batches = t_std.shape[1]
            for batch_index in range(n_batches):
                _, std = self._get_mean_std(metadata['category'][batch_index], t_std.device)

                new_std = t_std * std

        elif len(t_std.shape) == 2:
            # Single sample operation
            _, std = self._get_mean_std(metadata['category'], t_std.device)

            new_std = t_std * std
        else:
            raise AssertionError(f'Invalid bbox shape {t_std.shape}')

        return new_std

    def inverse_var(
        self,
        t_var: torch.Tensor,
        additional_data: Optional[TensorCollection] = None,
        shallow: bool = True
    ) -> TensorCollection:
        metadata, *other = additional_data
        new_var = None

        if len(t_var.shape) == 3:
            # Batch operation
            n_batches = t_var.shape[1]
            for batch_index in range(n_batches):
                _, std = self._get_mean_std(metadata['category'][batch_index], t_var.device)

                new_var = t_var * torch.square(std)

        elif len(t_var.shape) == 2:
            # Single sample operation
            _, std = self._get_mean_std(metadata['category'], t_var.device)

            new_var = t_var * torch.square(std)
        else:
            raise AssertionError(f'Invalid bbox shape {t_var.shape}')

        return new_var


class BBoxLogTransformRelativeToLastObs(InvertibleTransformWithVariance):
    """
    * For all observed coordinates:
    Y[i] = ln(X[-1] / X[i])
    Y[-1] is removed

    * For all unobserved coordinates:
    Y[i] = ln(X[i] / X[-1])
    """
    def __init__(self, clip_radius: Optional[float] = 1.0):
        super().__init__(name='log_relative_to_last_obs')
        self._min_clip = -clip_radius if clip_radius is not None else None
        self._max_clip = clip_radius if clip_radius is not None else None

    def apply(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        bbox_obs, bbox_unobs, ts_obs, *other = data
        last_obs = bbox_obs[-1:]

        if not shallow:
            bbox_obs = bbox_obs.clone()
            bbox_unobs = bbox_unobs.clone() if bbox_unobs is not None else None

        bbox_obs, ts_obs = bbox_obs[:-1], ts_obs[:-1]  # Last element becomes redundant
        bbox_obs = torch.nan_to_num(torch.log(last_obs.expand_as(bbox_obs) / bbox_obs), posinf=0, neginf=0)
        bbox_obs = torch.clip(bbox_obs, min=self._min_clip, max=self._max_clip) if self._min_clip is not None else bbox_obs
        if bbox_unobs is not None:
            bbox_unobs = torch.nan_to_num(torch.log(bbox_unobs / last_obs.expand_as(bbox_unobs)), posinf=0, neginf=0)
            bbox_unobs = torch.clip(bbox_unobs, min=self._min_clip, max=self._max_clip) if self._min_clip is not None else bbox_unobs

        return bbox_obs, bbox_unobs, ts_obs, *other

    def inverse(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        orig_bbox_obs, bbox_hat, *other = data
        last_obs = orig_bbox_obs[-1:]

        if not shallow:
            bbox_hat = bbox_hat.clone()

        bbox_hat = torch.exp(bbox_hat) * last_obs.expand_as(bbox_hat)

        return orig_bbox_obs, bbox_hat, *other

    def inverse_std(
        self,
        t_std: torch.Tensor,
        additional_data: Optional[TensorCollection] = None,
        shallow: bool = True
    ) -> TensorCollection:
        t_var = torch.square(t_std)
        var = self.inverse_var(t_var)
        std = torch.sqrt(var)
        return std  # It's assumed that last observed element has variance equal to 0

    def inverse_var(
        self,
        t_var: torch.Tensor,
        additional_data: Optional[TensorCollection] = None,
        shallow: bool = True
    ) -> TensorCollection:
        orig_bbox_obs, *other = additional_data
        last_obs = torch.square(orig_bbox_obs[-1:])
        return torch.exp(t_var) * last_obs.expand_as(t_var) # It's assumed that last observed element has variance equal to 0


class BBoxJackOfAllTradesTransform(InvertibleTransformWithVariance):
    def __init__(
        self,
        rel_mean: float,
        rel_std: float,
        diff_mean: float,
        diff_std: float
    ):
        super().__init__(name='jack_of_all_trades')
        self._rel_transform = BBoxStandardizedRelativeToLastObsTransform(mean=rel_mean, std=rel_std)
        self._diff_transform = BBoxStandardizedFirstOrderDifferenceTransform(mean=diff_mean, std=diff_std)

    def inverse(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        return self._rel_transform.inverse(data, shallow=shallow)

    def apply(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        bbox_obs, bbox_unobs, ts_obs, *other = data
        rel_bbox_obs, rel_bbox_unobs, rel_ts_obs, *_ = self._rel_transform([bbox_obs, bbox_unobs, ts_obs, None], shallow=False)
        diff_bbox_obs, diff_bbox_unobs, *_ = self._diff_transform([bbox_obs, bbox_unobs, ts_obs, None], shallow=False)
        bbox_obs = bbox_obs[1:]
        bbox_features = torch.cat([rel_bbox_obs, diff_bbox_obs, bbox_obs], dim=-1)

        return bbox_features, rel_bbox_unobs, rel_ts_obs, *other

    def inverse_std(self, t_std: torch.Tensor, additional_data: Optional[TensorCollection] = None, shallow: bool = True) -> TensorCollection:
        return self._rel_transform.inverse_std(t_std, additional_data=additional_data, shallow=shallow)

    def inverse_var(self, t_var: torch.Tensor, additional_data: Optional[TensorCollection] = None, shallow: bool = True) -> TensorCollection:
        return self._rel_transform.inverse_var(t_var, additional_data=additional_data, shallow=shallow)


# noinspection DuplicatedCode
def run_test_first_difference() -> None:
    bbox_obs = torch.randn(2, 2, 4)
    bbox_unobs = torch.randn(3, 2, 4)
    ts_obs = torch.randn(2, 2, 1)
    ts_unobs = torch.randn(3, 2, 1)
    first_diff = BboxFirstOrderDifferenceTransform()

    transformed_bbox_obs, transformed_bbox_unobs, *_ = \
        first_diff.apply([bbox_obs, bbox_unobs, ts_obs, ts_unobs, None], shallow=False)
    assert transformed_bbox_obs.shape == (1, 2, 4)
    assert transformed_bbox_unobs.shape == (3, 2, 4)

    _, inv_transformed_bbox_unobs, *_ = first_diff.inverse([bbox_obs, transformed_bbox_unobs, None])
    assert torch.abs(inv_transformed_bbox_unobs - bbox_unobs).sum().item() < 1e-3


def run_standardization() -> None:
    bbox_obs = torch.randn(2, 2, 4)
    bbox_unobs = torch.randn(3, 2, 4)
    ts_obs = torch.randn(2, 2, 1)
    ts_unobs = torch.randn(3, 2, 1)
    first_diff = BBoxStandardizationTransform(1, 2.0)

    _, transformed_bbox_unobs, *_ = \
        first_diff.apply([bbox_obs, bbox_unobs, ts_obs, ts_unobs, None], shallow=False)

    _, inv_transformed_bbox_unobs, *_ = first_diff.inverse([bbox_obs, transformed_bbox_unobs, None])
    assert torch.abs(inv_transformed_bbox_unobs - bbox_unobs).sum().item() < 1e-3


# noinspection DuplicatedCode
def run_test_standardized_first_difference() -> None:
    bbox_obs = torch.randn(2, 2, 4)
    bbox_unobs = torch.randn(3, 2, 4)
    ts_obs = torch.randn(2, 2, 1)
    ts_unobs = torch.randn(3, 2, 1)
    first_diff = BBoxStandardizedFirstOrderDifferenceTransform(1, 2.0)

    transformed_bbox_obs, transformed_bbox_unobs, *_ = \
        first_diff.apply([bbox_obs, bbox_unobs, ts_obs, ts_unobs, None], shallow=False)
    assert transformed_bbox_obs.shape == (1, 2, 4)
    assert transformed_bbox_unobs.shape == (3, 2, 4)

    _, inv_transformed_bbox_unobs, *_ = first_diff.inverse([bbox_obs, transformed_bbox_unobs, None])
    assert torch.abs(inv_transformed_bbox_unobs - bbox_unobs).sum().item() < 1e-3


# noinspection DuplicatedCode
def run_test_relative_to_last_obs() -> None:
    bbox_obs = torch.randn(2, 2, 4)
    bbox_unobs = torch.randn(3, 2, 4)
    ts_obs = torch.randn(2, 2, 1)
    ts_unobs = torch.randn(3, 2, 1)
    first_diff = BBoxRelativeToLastObsTransform()

    transformed_bbox_obs, transformed_bbox_unobs, *_ = \
        first_diff.apply([bbox_obs, bbox_unobs, ts_obs, ts_unobs, None], shallow=False)
    assert transformed_bbox_obs.shape == (1, 2, 4)
    assert transformed_bbox_unobs.shape == (3, 2, 4)

    _, inv_transformed_bbox_unobs, *_ = first_diff.inverse([bbox_obs, transformed_bbox_unobs, None])
    assert torch.abs(inv_transformed_bbox_unobs - bbox_unobs).sum().item() < 1e-3


# noinspection DuplicatedCode
def run_test_standardized_relative_to_last_obs() -> None:
    bbox_obs = torch.randn(2, 2, 4)
    bbox_unobs = torch.randn(3, 2, 4)
    ts_obs = torch.randn(2, 2, 1)
    ts_unobs = torch.randn(3, 2, 1)
    first_diff = BBoxStandardizedRelativeToLastObsTransform(1, 2.0)

    transformed_bbox_obs, transformed_bbox_unobs, *_ = \
        first_diff.apply([bbox_obs, bbox_unobs, ts_obs, ts_unobs, None], shallow=False)
    assert transformed_bbox_obs.shape == (1, 2, 4)
    assert transformed_bbox_unobs.shape == (3, 2, 4)

    _, inv_transformed_bbox_unobs, *_ = first_diff.inverse([bbox_obs, transformed_bbox_unobs, None])
    assert torch.abs(inv_transformed_bbox_unobs - bbox_unobs).sum().item() < 1e-3

# noinspection DuplicatedCode
def run_test_log_relative_to_last_obs() -> None:
    bbox_obs = torch.abs(torch.randn(2, 2, 4))
    bbox_unobs = torch.abs(torch.randn(3, 2, 4))
    ts_obs = torch.randn(2, 2, 1)
    ts_unobs = torch.randn(3, 2, 1)
    log_transform = BBoxLogTransformRelativeToLastObs(clip_radius=None)

    transformed_bbox_obs, transformed_bbox_unobs, *_ = \
        log_transform.apply([bbox_obs, bbox_unobs, ts_obs, ts_unobs, None], shallow=False)
    assert transformed_bbox_obs.shape == (1, 2, 4)
    assert transformed_bbox_unobs.shape == (3, 2, 4)

    _, inv_transformed_bbox_unobs, *_ = log_transform.inverse([bbox_obs, transformed_bbox_unobs, None])
    assert torch.abs(inv_transformed_bbox_unobs - bbox_unobs).sum().item() < 1e-3


def run_add_category_label_index() -> None:
    bbox_obs = torch.randn(2, 2, 4)
    bbox_unobs = torch.randn(3, 2, 4)
    ts_obs = torch.randn(2, 2, 1)
    ts_unobs = torch.randn(3, 2, 1)
    orig_bbox_obs = bbox_obs.clone()
    metadata = {'category': 'dog'}
    add_label = BBoxAddLabelTransform(
        token_to_index={
            '<unk>': 0,
            'dog': 1,
            'cat': 2
        },
        unknown_token='<unk>',
        add_unknown_token=True
    )

    transformed_bbox_obs, transformed_bbox_unobs, *_ = \
        add_label.apply([bbox_obs, bbox_unobs, ts_obs, ts_unobs, orig_bbox_obs, metadata, None], shallow=False)
    assert transformed_bbox_obs.shape == (2, 2, 5)
    assert transformed_bbox_unobs.shape == (3, 2, 4)

    _, inv_transformed_bbox_unobs, *_ = add_label.inverse([bbox_obs, transformed_bbox_unobs, None])
    assert torch.abs(inv_transformed_bbox_unobs - bbox_unobs).sum().item() < 1e-3


def run_test_jack_of_all_trades() -> None:
    bbox_obs = torch.randn(2, 2, 4)
    bbox_unobs = torch.randn(3, 2, 4)
    ts_obs = torch.randn(2, 2, 1)
    ts_unobs = torch.randn(3, 2, 1)
    jack_of_all_trades = BBoxJackOfAllTradesTransform(1.0, 2.0, 1.0, 2.0)

    transformed_bbox_obs, transformed_bbox_unobs, *_ = \
        jack_of_all_trades.apply([bbox_obs, bbox_unobs, ts_obs, ts_unobs, None], shallow=False)
    assert transformed_bbox_obs.shape == (1, 2, 12)
    assert transformed_bbox_unobs.shape == (3, 2, 4)

    _, inv_transformed_bbox_unobs, *_ = jack_of_all_trades.inverse([bbox_obs, transformed_bbox_unobs, None])
    assert torch.abs(inv_transformed_bbox_unobs - bbox_unobs).sum().item() < 1e-3


if __name__ == '__main__':
    run_test_first_difference()
    run_standardization()
    run_test_standardized_first_difference()
    run_test_relative_to_last_obs()
    run_test_standardized_relative_to_last_obs()
    run_add_category_label_index()
    run_test_log_relative_to_last_obs()
    run_test_jack_of_all_trades()
