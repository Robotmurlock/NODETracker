"""
Model factory method
"""
import enum
from typing import Union, Optional

from pytorch_lightning import LightningModule

from nodetracker.datasets.transforms import InvertibleTransform, InvertibleTransformWithVariance
from nodetracker.node.core.odevae import LightningODEVAE
from nodetracker.node.kalman_filter import TorchConstantVelocityODKalmanFilter
from nodetracker.node.nlp import LightningCategoryRNNODE
from nodetracker.node.odernn import (
    LightningODERNN,
    LightningODERNNVAE,
    LightningRNNODE,
    LightningMLPODE,
    LightningComposeRNNODE,
    LightningNODEFilterModel
)
from nodetracker.node.utils import LightningTrainConfig
from nodetracker.node.utils.training import LightningModuleForecaster
from nodetracker.np import LightningBaselineCNP, LightningBaselineAttnCNP, LightningBaselineRNNCNP, LightningRNNCNPFilter
from nodetracker.standard.flow import LightningSingleStepFlowRNN
from nodetracker.standard.mlp import LightningMLPForecaster
from nodetracker.standard.rnn import LightningRNNSeq2Seq, LightningARRNN, LightningSingleStepRNN, LightningRNNFilterModel
from nodetracker.standard.trainable_kalman_filter import LightningAdaptiveKalmanFilter


class ModelType(enum.Enum):
    """
    Enumerated implemented architectures
    """
    # NODE
    ODEVAE = 'odevae'
    ODERNN = 'odernn'
    ODERNNVAE = 'odernnvae'
    RNNODE = 'rnnode'
    CATEGORY_RNNODE = 'category_rnnode'
    COMPOSE_RNNODE = 'compose_rnnode'
    MLPODE = 'mlpode'
    NODE_FILTER = 'node_filter'

    # RNN
    ARRNN = 'arrnn'
    SINGLE_STEP_RNN = 'single-step-rnn'
    SINGLE_STEP_FLOW_RNN = 'single-step-flow-rnn'
    RNN = 'rnn'
    RNN_FILTER = 'rnn_filter'

    # MLP
    MLP = 'mlp'

    # KF
    KALMAN_FILTER = 'kf'
    TAKF = 'takf'

    # NP
    BASELINE_CNP = 'baseline-cnp'
    BASELINE_ATTN_CNP = 'baseline-attn-cnp'
    BASELINE_RNN_CNP = 'baseline-rnn-cnp'
    RNN_CNP_FILTER = 'rnn-cnp-filter'

    @classmethod
    def from_str(cls, value: str) -> 'ModelType':
        for v in cls:
            if v.value.lower() == value.lower():
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
    n_train_steps: Optional[int] = None,
    train_params: Optional[dict] = None,
    transform_func: Optional[Union[InvertibleTransform, InvertibleTransformWithVariance]] = None
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
        n_train_steps: Number of train steps
        train_params: Parameters for model training
        transform_func: Transform function (applied before loss)
    Returns:
        Model
    """
    if isinstance(model_type, str):
        model_type = ModelType.from_str(model_type)

    catalog = {
        ModelType.ODEVAE: LightningODEVAE,
        ModelType.ODERNN: LightningODERNN,
        ModelType.RNNODE: LightningRNNODE,
        ModelType.NODE_FILTER: LightningNODEFilterModel,
        ModelType.ODERNNVAE: LightningODERNNVAE,
        ModelType.ARRNN: LightningARRNN,
        ModelType.RNN: LightningRNNSeq2Seq,
        ModelType.SINGLE_STEP_RNN: LightningSingleStepRNN,
        ModelType.SINGLE_STEP_FLOW_RNN: LightningSingleStepFlowRNN,
        ModelType.RNN_FILTER: LightningRNNFilterModel,
        ModelType.MLP: LightningMLPForecaster,
        ModelType.MLPODE: LightningMLPODE,
        ModelType.KALMAN_FILTER: TorchConstantVelocityODKalmanFilter,
        ModelType.TAKF: LightningAdaptiveKalmanFilter,
        ModelType.CATEGORY_RNNODE: LightningCategoryRNNODE,
        ModelType.COMPOSE_RNNODE: LightningComposeRNNODE,
        ModelType.BASELINE_CNP: LightningBaselineCNP,
        ModelType.BASELINE_ATTN_CNP: LightningBaselineAttnCNP,
        ModelType.BASELINE_RNN_CNP: LightningBaselineRNNCNP,
        ModelType.RNN_CNP_FILTER: LightningRNNCNPFilter
    }

    model_cls = catalog[model_type]

    if not model_type.trainable:
        return model_cls(**params)

    if n_train_steps is not None:
        train_params['n_train_steps'] = n_train_steps

    train_config = LightningTrainConfig(**train_params) if train_params is not None else None
    if checkpoint_path is None and train_config is None:
        # It does not make sense to use trainable model with no train parameters and no checkpoint path
        raise ValueError('Train config and checkpoint path can\'t be both None for trainable models!')

    if checkpoint_path is not None:
        return model_cls.load_from_checkpoint(
            **params,
            checkpoint_path=checkpoint_path,
            train_config=train_config,
            transform_func=transform_func
        )

    return model_cls(
        **params,
        train_config=train_config,
        transform_func=transform_func
    )
