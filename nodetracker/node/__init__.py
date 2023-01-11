"""
Set of different NODE implementations
"""
from nodetracker.node.generative_latent_time_series_model import ODEVAE, LightningODEVAE
from nodetracker.node.ode_rnn import ODERNNVAE, LightningODERNNVAE
from nodetracker.node.kalman_filter import TorchConstantVelocityODKalmanFilter
from nodetracker.node.factory import load_or_create_model, ModelType
from nodetracker.node.trajectory_filter import BBoxTrajectoryForecaster
