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
    BBoxStandardizedRelativeToLastObsTransform,
    BBoxRelativeToLastObsWithPadTransform,
    BBoxCompositeTransform,
    BBoxAddLabelTransform,
    BBoxCategoryStandardizationTransform,
    BBoxLogTransformRelativeToLastObs,
    BBoxJackOfAllTradesTransform,
    BBoxNormalizeToLastObsTransform,
    BBoxStandardizedNormalizeToLastObsTransform,
    BBoxNormalizedDifferencesTransform,
    BBoxStandardizedNormalizedDifferencesTransform
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
        'composite': BBoxCompositeTransform,
        'first_difference': BboxFirstOrderDifferenceTransform,
        'standardization': BBoxStandardizationTransform,
        'standardized_first_difference': BBoxStandardizedFirstOrderDifferenceTransform,
        'relative_to_last_obs': BBoxRelativeToLastObsTransform,
        'standardized_relative_to_last_obs': BBoxStandardizedRelativeToLastObsTransform,
        'standardized_relative_to_last_obs_with_pad': BBoxRelativeToLastObsWithPadTransform,
        'add_label': BBoxAddLabelTransform,
        'standardization_by_category': BBoxCategoryStandardizationTransform,
        'log_relative_to_last_obs': BBoxLogTransformRelativeToLastObs,
        'jack_of_all_trades': BBoxJackOfAllTradesTransform,
        'normalize_to_last_obs': BBoxNormalizeToLastObsTransform,
        'standardized_normalize_to_last_obs': BBoxStandardizedNormalizeToLastObsTransform,
        'normalized_differences': BBoxNormalizedDifferencesTransform,
        'standardized_normalized_differences': BBoxStandardizedNormalizedDifferencesTransform
    }

    cls = catalog[name]

    if name == 'composite':
        params['transforms'] = [transform_factory(child['name'], child['params']) for child in params['transforms']]

    return cls(**params)
