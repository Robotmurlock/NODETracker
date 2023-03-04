"""
Model factory method
"""
import enum
from typing import Union, Optional

from pytorch_lightning import LightningModule

from nodetracker.node.core.odevae import LightningODEVAE
from nodetracker.node.kalman_filter import TorchConstantVelocityODKalmanFilter
from nodetracker.node.odernn import LightningODERNN, LightningODERNNVAE
from nodetracker.node.standard.simple import LightningRNNSeq2Seq
from nodetracker.node.standard.mlp import LightningMLPForecaster
from nodetracker.node.utils.training import LightningModuleForecaster
from nodetracker.node.utils import LightningTrainConfig


class ModelType(enum.Enum):
    """
    Enumerated implemented architectures
    """
    ODEVAE = 'odevae'
    ODERNN = 'odernn'
    ODERNNVAE = 'odernnvae'
    RNN = 'rnn'
    MLP = 'mlp'
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


def load_or_create_model(
    model_type: Union[ModelType, str],
    params: dict,
    checkpoint_path: Optional[str] = None,
    train_params: Optional[dict] = None
) -> Union[LightningModuleForecaster, LightningModule]:
    """
    Loads trained (if given checkpoint path) or creates new model given name and parameters.
    If model is trainable (check ModelType) then it can use train config. Otherwise, it can be loaded from checkpoint.
    Parameters combinations:
    - checkpoint_path is None and train_params is None - not allowed
    - checkpoint_path is None and train_params is not None - model is used for training from scratch
    - checkpoint_path is not None and train_params is None - model is used for inference
    - checkpoint_path is None and train_params is None - model is used for training from checkpoint (continue training)
    Args:
        model_type: Model type
        params: Model parameters'
        checkpoint_path: Load pretrained model
        train_params: Parameters for model training
    Returns:
        Model
    """
    if isinstance(model_type, str):
        model_type = ModelType.from_str(model_type)

    catalog = {
        ModelType.ODEVAE: LightningODEVAE,
        ModelType.ODERNN: LightningODERNN,
        ModelType.ODERNNVAE: LightningODERNNVAE,
        ModelType.RNN: LightningRNNSeq2Seq,
        ModelType.MLP: LightningMLPForecaster,
        ModelType.KALMAN_FILTER: TorchConstantVelocityODKalmanFilter
    }

    model_cls = catalog[model_type]

    if not model_type.trainable:
        return model_cls(**params)

    train_config = LightningTrainConfig(**train_params) if train_params is not None else None
    if checkpoint_path is None and train_config is None:
        # It does not make sense to use trainable model with no train parameters and no checkpoint path
        raise ValueError('Train config and checkpoint path can\'t be both None for trainable models!')

    if checkpoint_path is not None:
        return model_cls.load_from_checkpoint(checkpoint_path=checkpoint_path, **params, train_config=train_config)

    return model_cls(**params, train_config=train_config)
