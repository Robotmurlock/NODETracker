from nodetracker.datasets.transforms import (
    InvertibleTransform,
    IdentityTransform,
    BboxFirstOrderDifferenceTransform,
    BBoxStandardizationTransform,
    BBoxStandardizedFirstOrderDifferenceTransform
)


def transform_factory(name: str, params: dict) -> InvertibleTransform:
    catalog = {
        'identity': IdentityTransform,
        'first_difference': BboxFirstOrderDifferenceTransform,
        'standardization': BBoxStandardizationTransform,
        'standardized_first_difference': BBoxStandardizedFirstOrderDifferenceTransform
    }

    cls = catalog[name]
    return cls(**params)
