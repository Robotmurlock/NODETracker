from abc import ABC, abstractmethod
from typing import Collection

class Transform(ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """
        Returns:
            Transformation name
        """
        return self._name

    @abstractmethod
    def apply(self, data: Collection, *args, **kwargs) -> Collection:
        pass

    def __call__(self, data: Collection, *args, **kwargs) -> Collection:
        return self.apply(data)

class InvertibleTransform(Transform, ABC):
    def __init__(self, name: str):
        super().__init__(name=name)
    @abstractmethod
    def inverse(self, data: Collection, *args, **kwargs) -> Collection:
        pass


class IdentityTransform(InvertibleTransform):
    def __init__(self):
        super().__init__(name='identity')
    def apply(self, data: Collection, *args, **kwargs) -> Collection:
        return data
    def inverse(self, data: Collection, *args, **kwargs) -> Collection:
        return data
