"""
ODE Solver implementations. Supports:
- Euler Method
"""
from abc import ABC, abstractmethod
from typing import Union, Callable, Tuple, Optional

import torch
from torch import nn
import matplotlib.pyplot as plt


class ODESolver(ABC):
    """
    Interface definition for ODESolver
    """
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


class RungeKuttaMethod(ODESolver, ABC):
    """
    Baseline for RungeKutta methods
    https://en.wikipedia.org/wiki/Runge–Kutta_methods
    """
    def __init__(self, max_step_size: float):
        """
        Simplest ODESolver

        Args:
            max_step_size: Max step size (higher step size ~ lower precision)
        """
        super().__init__()

        self._max_step_size = max_step_size

    @abstractmethod
    def _estimate_increment(self, f: nn.Module, z: torch.Tensor, t: torch.Tensor, h: float) -> torch.Tensor:
        """
        RungeKutta estimate increment implementation (depends on the order of RK method).

        Args:
            f: State change function
            z: State
            t: Time
            h: step size

        Returns:
            Estimated step increment
        """
        pass

    # noinspection DuplicatedCode
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
            z = z + self._estimate_increment(f, z, t, h) * h
            t = t + h

            if return_all_states:
                zs.append(z)
                ts.append(t)

        if return_all_states:
            zs, ts = torch.stack(zs), torch.stack(ts)
            return zs, ts

        return z, t  # only last state


class EulerMethod(RungeKuttaMethod):
    """
    Implementation of Euler Method algorithm (RungeKutta 1st order).
    https://en.wikipedia.org/wiki/Euler_method
    """
    def _estimate_increment(self, f: nn.Module, z: torch.Tensor, t: torch.Tensor, h: float) -> torch.Tensor:
        _ = h  # void(h)
        return f(z, t)


class RK4(RungeKuttaMethod):
    """
    RungeKutta 4th order.
    """

    def _estimate_increment(self, f: nn.Module, z: torch.Tensor, t: torch.Tensor, h: float) -> torch.Tensor:
        k1 = f(z, t)

        z2 = z + h*k1/2
        t2 = t + h/2
        k2 = f(z2, t2)

        z3 = z + h*k2/2
        t3 = t + h/2
        k3 = f(z3, t3)

        z4 = z + h*k3
        t4 = t + h
        k4 = f(z4, t4)

        return (k1 + 2*k2 + 2*k3 + k4) / 6
