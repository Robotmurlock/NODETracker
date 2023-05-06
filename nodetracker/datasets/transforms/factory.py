"""
Transform factory method.
"""
from nodetracker.datasets.transforms import (
    InvertibleTransform,
    InvertibleTransformWithVariance,
    IdentityTransform,
    BboxFirstOrderDifferenceTransform,
    BBoxStandardizationTransform,
    BBoxStandardizedFirstOrderDifferenceTransform,
    BBoxRelativeToLastObsTransform,
    BBoxStandardizedRelativeToLastObsTransform
)
from typing import Union


def transform_factory(name: str, params: dict) -> Union[InvertibleTransform, InvertibleTransformWithVariance]:
    """
    Create transform object based on given name and constructor parameters.

    Args:
        name: Transform name
        params: Transform parameters

    Returns:
        Transform object
    """
    catalog = {
        'identity': IdentityTransform,
        'first_difference': BboxFirstOrderDifferenceTransform,
        'standardization': BBoxStandardizationTransform,
        'standardized_first_difference': BBoxStandardizedFirstOrderDifferenceTransform,
        'relative_to_last_obs': BBoxRelativeToLastObsTransform,
        'standardized_relative_to_last_obs': BBoxStandardizedRelativeToLastObsTransform
    }

    cls = catalog[name]
    return cls(**params)
