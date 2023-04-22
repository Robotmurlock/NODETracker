"""
Create split index for LaSOT dataset
"""
import argparse
import os
from typing import Dict, List
from collections import defaultdict

import yaml

from nodetracker.utils import file_system

DatasetIndex = Dict[str, List[str]]
SplitIndex = Dict[str, List[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='LaSOT dataset split index.')
    parser.add_argument('--path', type=str, required=True, help='Dataset path (required for indexing)')
    parser.add_argument('--train-val-ratio', type=float, required=True, help='Train to val ratio. Interval: [0, 1].')
    parser.add_argument('--version', type=str, required=False, help='Split version.')
    return parser.parse_args()


def index_dataset(path: str) -> DatasetIndex:
    """
    Indexes dataset in format:

    {category-1}/
        {category-1-sequence-1}
        {category-1-sequence-2}
        ...
    {category-2}/
        {category-2-sequence-1}
        {category-2-sequence-2}
        ...
    ...

    Args:
        path: Dataset path

    Returns:
        Dataset index
    """
    index: DatasetIndex = defaultdict(list)

    sequences = file_system.listdir(path)
    for sequence in sequences:
        category, _ = sequence.split('-')
        index[category].append(sequence)

    index = {k: sorted(v, key=lambda x: int(x.split('-')[-1])) for k, v in index.items()}
    return dict(index)


def perform_split(index: DatasetIndex, train_val_ratio: float) -> SplitIndex:
    """
    Performs split on indexed data.

    Args:
        index: Dataset index
        train_val_ratio: Train to val ratio for split

    Returns:
        Train and Validation index (split) - mapping
    """
    split_index: SplitIndex = defaultdict(list)

    for category, sequences in index.items():
        n_seq = len(sequences)
        n_train_seq = round(n_seq * train_val_ratio)
        n_val_seq = n_seq - n_train_seq
        assert n_train_seq > 0, f'There are 0 training sequences for category "{category}" with {n_seq} sequences!'
        assert n_val_seq > 0, f'There are 0 training sequences for category "{category}" with {n_seq} sequences!'

        split_index['train'].extend(sequences[:n_train_seq])
        split_index['val'].extend(sequences[n_train_seq:])

    return dict(split_index)


def save_split_index(split_index: SplitIndex, path: str, version: str) -> None:
    """
    Saves split index.

    Args:
        split_index: Split index
        path: Dataset path
        version: Split index version
    """
    split_index_path = os.path.join(path, f'.split_index_{version}.yaml')
    if os.path.exists(split_index_path):
        raise FileExistsError(f'File "{split_index}" already exists!')

    with open(split_index_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(split_index, f)


def main(cfg: argparse.Namespace) -> None:
    index = index_dataset(cfg.path)
    split_index = perform_split(index, cfg.train_val_ratio)
    save_split_index(split_index, cfg.path, cfg.version)


if __name__ == '__main__':
    main(parse_args())
