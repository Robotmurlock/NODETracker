"""
Implementation of training augmentations
"""
from nodetracker.datasets.augmentations.trajectory import (
    TrajectoryAugmentation,
    IdentityAugmentation,
    DetectorNoiseAugmentation,
    CompositionAugmentation,
    create_identity_augmentation_config
)
