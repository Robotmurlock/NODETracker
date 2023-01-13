"""
Model factory method
"""
import enum
from typing import Union, Optional

from nodetracker.node.node_trainer import LightningODEVAE
from nodetracker.node.kalman_filter import TorchConstantVelocityODKalmanFilter
from nodetracker.node.trajectory_filter import TrajectoryFilter


class ModelType(enum.Enum):
    """
    Enumerated implemented architectures
    """
    ODEVAE = 'odevae'
    ODERNNVAE = 'odernnvae'
    KALMAN_FILTER = 'kf'

    @classmethod
    def from_str(cls, value: str) -> 'ModelType':
        for v in cls:
            if v.value == value:
                return v

        raise ValueError(f'Can\'t create ModelType from "{value}". Possible values: {list(cls)}')

    @property
    def trainable(self) -> bool:
        """
        Returns: Should the model be trained before performing inference
        """
        return self.value not in [ModelType.KALMAN_FILTER.value]

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
        ModelType.ODERNNVAE: LightningODEVAE,
        ModelType.KALMAN_FILTER: TorchConstantVelocityODKalmanFilter
    }

    if model_type in [ModelType.ODEVAE, ModelType.ODERNNVAE]:
        # Not the cleanest code but necesary in order for ODEVAE and ODERNNVAE to use same PL trainer
        if 'model_name' in params:
            raise KeyError('Unexpected "model_name" found in model parameters!')
        params['model_name'] = model_type.value

    if checkpoint_path is None:
        if not model_type.trainable:
            raise ValueError('Models that are not trainable can\'t be loaded from checkpoint!')
        return catalog[model_type](**params)

    return catalog[model_type].load_from_checkpoint(checkpoint_path, **params)
