"""
Training Meters
"""
from collections import defaultdict
from typing import Tuple, Union, Iterable

import torch


class UnknownMetricException(KeyError):
    """
    No metric value was pushed for this metric name
    """
    pass


class MetricMeter:
    """
    Metric/Loss Meter for training
    """
    def __init__(self):
        self._history = defaultdict(list)

    def push(self, name: str, value: Union[torch.Tensor, float]) -> None:
        """
        Pushes new metric value. List of these values are later used to calculate average metric value
        by using `get()` or `get_all()` method.

        Args:
            name: Metric name
            value: Metric value
        """
        self._history[name].append(value)

    def get(self, name: str) -> Union[torch.Tensor, float]:
        """
        Calculates average over all pushed metric values.

        Args:
            name: Name of the metric

        Returns:
            Metric score
        """
        if name not in self._history:
            raise UnknownMetricException(f'Metric name "{name}" not found. '
                                         f'Known metrics: {list(self._history.keys())}.')

        values = self._history[name]
        return sum(values) / len(values)

    def get_all(self, flush: bool = True) -> Iterable[Tuple[str, Union[torch.Tensor, float]]]:
        """
        Calculates averages over all pushed metric values for every pushed metrics.
        Optionally deletes all data if flush is activated (default).

        Args:
            flush: Performs flush at the end

        Returns:
            Iterator: name of the metric, metric score
        """
        for name in self._history:
            yield name, self.get(name)

        if flush:
            self.flush()

    def flush(self) -> None:
        """
        Deletes all pushed data.
        """
        self._history = defaultdict(list)
