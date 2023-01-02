from abc import ABC, abstractmethod
from typing import Union, Callable, Tuple

import torch
import torch.nn as nn


class ODESolver(ABC):
    @abstractmethod
    def solve(
            self,
            z0: torch.Tensor,
            t0: torch.Tensor,
            t1: torch.Tensor,
            f: Union[nn.Module, Callable],
            return_all_states: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solves ODE

        Args:
            z0: Start state ~ z(t0)
            t0: Start time
            t1: End time
            f: State change function (slope)
            return_all_states: Return all intermediate states

        Returns:
            End state ~ z(t1) and end time point ~ t1 if return_all_states == False
            else return all states and time points
        """
        pass

    def __call__(
            self,
            z0: torch.Tensor,
            t0: torch.Tensor,
            t1: torch.Tensor,
            f: Union[nn.Module, Callable],
            return_all_states: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        solve() method wrapper
        """
        return self.solve(z0, t0, t1, f, return_all_states=return_all_states)


class EulerMethod(ODESolver):
    def __init__(self, max_step_size: float):
        """
        Simplest ODESolver

        Args:
            max_step_size: Max step size (higher step size ~ lower precision)
        """
        super().__init__()

        self._max_step_size = max_step_size

    def solve(
            self,
            z0: torch.Tensor,
            t0: torch.Tensor,
            t1: torch.Tensor,
            f: Union[nn.Module, Callable],
            return_all_states: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # noinspection PyTypeChecker
        n_steps = int(torch.ceil(torch.abs(t1 - t0) / self._max_step_size).max().item())
        h = (t1 - t0) / n_steps  # step size

        # Initial state
        t = t0
        z = z0

        zs = [z0]
        ts = [t0]
        for _ in range(n_steps):
            z = z + f(z, t) * h
            t = t + h

            if return_all_states:
                zs.append(z)
                ts.append(t)

        if return_all_states:
            zs, ts = torch.stack(zs), torch.stack(ts)
            return zs, ts

        return z, t  # only last state


DefaultODESolver = EulerMethod(0.05)  # Default ODESolver

