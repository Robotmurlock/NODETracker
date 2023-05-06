"""
Implementations of data transformations.
"""
from abc import ABC, abstractmethod
from typing import Collection, Union, Optional

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


class InvertibleTransformWithVariance(InvertibleTransform):
    """
    Extended InvertibleTransform with inverse for std/var
    """
    def __init__(self, name: str):
        super().__init__(name=name)

    @abstractmethod
    def inverse_std(self, t_std: torch.Tensor, additional_data: Optional[TensorCollection] = None, shallow: bool = True) -> TensorCollection:
        """
        Performs "inverse" transformation on std given the transformed data.

        Args:
            t_std: Transformed std
            additional_data: Data required to perform inverse std operation
            shallow: Take shallow copy of data (may cause side effects)

        Returns:
            "Untransformed" std
        """
        pass

    @abstractmethod
    def inverse_var(self, t_var: torch.Tensor, additional_data: Optional[TensorCollection] = None, shallow: bool = True) -> TensorCollection:
        """
        Performs "inverse" transformation on variance given the transformed data.

        Args:
            t_var: Transformed variance
            additional_data: Data required to perform inverse std operation
            shallow: Take shallow copy of data (may cause side effects)

        Returns:
            "Untransformed" variance
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
