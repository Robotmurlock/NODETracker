"""
Set of different NODE implementations
"""
from nodetracker.node.node_trainer import LightningODEVAE
from nodetracker.node.generative_latent_time_series_model import ODEVAE
from nodetracker.node.ode_rnn import ODERNNVAE
from nodetracker.node.kalman_filter import TorchConstantVelocityODKalmanFilter
from nodetracker.node.factory import load_or_create_model, ModelType
