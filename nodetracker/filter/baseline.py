from typing import Tuple

import torch

from nodetracker.filter.base import StateModelFilter, State


class BaselineFilter(StateModelFilter):
    """
    Baseline filter trusts fully detector and is completely certain in detector accuracy.
    """
    def __init__(self, det_uncertainty = 1e-6):
        self._det_uncertainty = det_uncertainty

    def initiate(self, measurement: torch.Tensor) -> State:
        return measurement

    def predict(self, state: State) -> State:
        measurement = state
        return measurement

    def multistep_predict(self, state: State, n_steps: int) -> State:
        measurement = state
        return torch.stack([measurement for _ in range(n_steps)])

    def update(self, state: State, measurement: torch.Tensor) -> State:
        _ = state  # ignore previous state
        return measurement

    def missing(self, state: State) -> State:
        return state

    def project(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        measurement = state
        det_mat = torch.ones_like(measurement) * self._det_uncertainty
        return state, det_mat

    def singlestep_to_multistep_state(self, state: State) -> State:
        return state.unsqueeze(0)
