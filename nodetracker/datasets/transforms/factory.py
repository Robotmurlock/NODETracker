from nodetracker.datasets.transforms import BboxFirstDifferenceTransform, InvertibleTransform, IdentityTransform


def transform_factory(name: str, params: dict) -> InvertibleTransform:
    catalog = {
        'first_difference': BboxFirstDifferenceTransform,
        'identity': IdentityTransform
    }

    cls = catalog[name]
    return cls(**params)
