"""
Torch dataset support. Any Dataset that implements `TrajectoryDataset` interface can be used for training and evaluation.
"""
from nodetracker.datasets.torch.core import TorchTrajectoryDataset, TrajectoryDataset
from nodetracker.datasets.torch.testing import run_dataset_test
