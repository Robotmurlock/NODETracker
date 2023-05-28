"""
Defines interface for motion model filter (predict - update)
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch

State = Optional[Tuple[torch.Tensor, ...]]


class StateModelFilter(ABC):
    """
    Defines interface for motion model filter (predict - update)
    """
    @abstractmethod
    def initiate(self, measurement: torch.Tensor) -> State:
        """
        Initializes model state with initial measurement.

        Args:
            measurement: Starting measurement

        Returns:
            Initial state
        """
        pass

    @abstractmethod
    def predict(self, state: State) -> State:
        """
        Predicts prior state.

        Args:
            state: Previous posterior state.

        Returns:
            Prior state (prediction)
        """
        pass

    @abstractmethod
    def multistep_predict(self, state: State, n_steps: int) -> State:
        """
        Estimates prior state for multiple steps ahead

        Args:
            state: Current state
            n_steps: Number of prediction steps

        Returns:
            Estimations for n_steps ahead
        """
        pass

    @abstractmethod
    def update(self, state: State, measurement: torch.Tensor) -> State:
        """
        Updates state model based on new measurement.

        Args:
            state: Prior state.
            measurement: New measurement

        Returns:
            Posterior state.
        """
        pass

    @abstractmethod
    def missing(self, state: State) -> State:
        """
        Update state when measurement is missing (replacement for update function for that case).

        Args:
            state: State (prior)

        Returns:
            Updated state
        """

    @abstractmethod
    def project(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects state to the measurement space (estimation with uncertainty)

        Args:
            state: Model state

        Returns:
            Mean, Covariance
        """
        pass
