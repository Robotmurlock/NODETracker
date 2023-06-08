"""
ODE Solver implementations. Supports:
- Euler Method
"""
from abc import ABC, abstractmethod
from typing import Union, Callable, Tuple

import torch
from torch import nn

from nodetracker.node.core.solver.core import ODESolver


class RungeKuttaMethodWithGlobalState(ODESolver, ABC):
    """
    Baseline for RungeKutta methods
    https://en.wikipedia.org/wiki/Runge–Kutta_methods
    """
    def __init__(self, max_step_size: float, split_dim: int = 1):
        """
        Simplest ODESolver

        Args:
            max_step_size: Max step size (higher step size ~ lower precision)
        """
        super().__init__()

        self._max_step_size = max_step_size
        self._split_dim = split_dim

    @abstractmethod
    def _estimate_increment(self, f: nn.Module, d: torch.Tensor, g: torch.Tensor, t: torch.Tensor, h: float) -> torch.Tensor:
        """
        RungeKutta estimate increment implementation (depends on the order of RK method).

        Args:
            f: State change function
            d: Dynamic state
            g: Static (global) state
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
            size = z.shape[self._split_dim] // 2
            d, g = torch.split(z, size, dim=self._split_dim)
            d = d + self._estimate_increment(f, d, g, t, h) * h
            z = torch.cat([d, g], dim=self._split_dim)
            t = t + h

            if return_all_states:
                zs.append(z)
                ts.append(t)

        if return_all_states:
            zs, ts = torch.stack(zs), torch.stack(ts)
            return zs, ts

        return z, t  # only last state


class EulerMethodWithGlobalState(RungeKuttaMethodWithGlobalState):
    """
    Implementation of Euler Method algorithm (RungeKutta 1st order).
    https://en.wikipedia.org/wiki/Euler_method
    """
    def _estimate_increment(self, f: nn.Module, d: torch.Tensor, g: torch.Tensor, t: torch.Tensor, h: float) -> torch.Tensor:
        _ = h  # void(h)
        return f(d, g, t)


class RK4WithGlobalState(RungeKuttaMethodWithGlobalState):
    """
    RungeKutta 4th order.
    """
    def _estimate_increment(self, f: nn.Module, d: torch.Tensor, g: torch.Tensor, t: torch.Tensor, h: float) -> torch.Tensor:
        k1 = f(d, g, t)

        d2 = d + h*k1/2
        t2 = t + h/2
        k2 = f(d2, g, t2)

        d3 = d + h*k2/2
        t3 = t + h/2
        k3 = f(d3, g, t3)

        d4 = d + h*k3
        t4 = t + h
        k4 = f(d4, g, t4)

        return (k1 + 2*k2 + 2*k3 + k4) / 6

