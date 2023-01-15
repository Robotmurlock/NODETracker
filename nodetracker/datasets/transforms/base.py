"""
Data transformations
"""
from abc import ABC, abstractmethod
from typing import Collection, Union

import torch


TensorCollection = Union[torch.Tensor, Collection[torch.Tensor]]


class Transform(ABC):
    """
    Maps data with implemented transformation.
    """
    def __init__(self, name: str):
        """
        Args:
            name: Transformation name.
        """
        self._name = name

    @property
    def name(self) -> str:
        """
        Returns:
            Transformation name
        """
        return self._name

    @abstractmethod
    def apply(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        """
        Perform transformation on given raw data.

        Args:
            data: Raw data
            shallow: Take shallow copy of data (may cause side effects but faster in general)

        Returns:
            Transformed data
        """
        pass

    def __call__(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        return self.apply(data)

class InvertibleTransform(Transform, ABC):
    """
    Transform that also implements `inverse` method.
    """
    def __init__(self, name: str):
        super().__init__(name=name)
    @abstractmethod
    def inverse(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        """
        Performs inverse transformation on given transformed data.

        Args:
            data: Transformed data
            shallow: Take shallow copy of data (may cause side effects)

        Returns:
            "Untransformed" data
        """
        pass


class IdentityTransform(InvertibleTransform):
    """
    Transformation neutral operator.
    """
    def __init__(self):
        super().__init__(name='identity')
    def apply(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        return data
    def inverse(self, data: TensorCollection, shallow: bool = True) -> TensorCollection:
        return data
