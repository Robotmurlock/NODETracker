"""
Model factory method
"""
import enum
from typing import Union, Optional

from nodetracker.node.generative_latent_time_series_model import LightningODEVAE
from nodetracker.node.kalman_filter import TorchConstantVelocityODKalmanFilter
from nodetracker.node.trajectory_filter import TrajectoryFilter


class ModelType(enum.Enum):
    ODEVAE = 'odevae'
    KALMAN_FILTER = 'kf'

    @classmethod
    def from_str(cls, value: str) -> 'ModelType':
        for v in cls:
            if v.value == value:
                return v

        raise ValueError(f'Can\'t create ModelType from "{value}". Possible values: {[v for v in cls]}')

    @property
    def trainable(self) -> bool:
        """
        Returns: Should the model be trained before performing inference
        """
        return self.value in [ModelType.ODEVAE.value]

def load_or_create_model(model_type: Union[ModelType, str], params: dict, checkpoint_path: Optional[str] = None) \
        -> Union['TrajectoryFilter', LightningODEVAE]:
    """
    Loads trained (if given checkpoint path) or creates new model given name and parameters

    Args:
        model_type: Model type
        params: Model parameters
        checkpoint_path: Load pretrained model

    Returns:
        Model
    """
    if isinstance(model_type, str):
        model_type = ModelType.from_str(model_type)

    catalog = {
        ModelType.ODEVAE: LightningODEVAE,
        ModelType.KALMAN_FILTER: TorchConstantVelocityODKalmanFilter
    }

    if checkpoint_path is None:
        return catalog[model_type](**params)
    return catalog[model_type].load_from_checkpoint(checkpoint_path, **params)
