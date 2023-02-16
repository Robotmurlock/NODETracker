"""
Set of different NODE implementations
"""
from nodetracker.node.factory import load_or_create_model, ModelType
from nodetracker.node.core.odevae import ODEVAE, LightningODEVAE
from nodetracker.node.kalman_filter import TorchConstantVelocityODKalmanFilter
from nodetracker.node.odernn.odernnvae import ODERNNVAE, LightningODERNNVAE
from nodetracker.node.odernn.odernn import ODERNN, LightningODERNN
from nodetracker.node.trajectory_filter import BBoxTrajectoryForecaster
from nodetracker.node.utils import LightningTrainConfig
