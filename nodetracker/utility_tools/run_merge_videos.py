"""
Tool for horizontal video merging.
"""
import argparse
import logging

import numpy as np
from tqdm import tqdm

from nodetracker.library.cv import MP4Reader, MP4Writer
from nodetracker.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    """
    Merge two videos horizontally - configuration.

    Returns:
        Parsed config.
    """
    parser = argparse.ArgumentParser(description='MergeVideo')
    parser.add_argument('--left', type=str, required=True, help='Path to the left video.')
    parser.add_argument('--right', type=str, required=True, help='Path to the right video.')
    parser.add_argument('--merged', type=str, required=True, help='Path to the output (merged) video.')
    parser.add_argument('--fps', type=int, required=False, default=30, help='Output video fps')
    return parser.parse_args()


def main(cfg: argparse.Namespace):
    with MP4Reader(cfg.left) as left_reader, \
            MP4Reader(cfg.right) as right_reader, \
            MP4Writer(cfg.merged, cfg.fps) as writer:

        with tqdm(zip(left_reader.iter(), right_reader.iter()), unit='frame', desc='Merging') as pbar:
            for (_, left_frame), (_, right_frame) in pbar:
                assert left_frame.shape == right_frame.shape, \
                    f'Video resolutions do not match: {left_frame.shape} != {right_frame.shape}'

                h, w, _ = left_frame.shape

                merged_frame = np.zeros(shape=(h, 2 * w, 3), dtype=np.uint8)
                merged_frame[:, :w, :] = left_frame
                merged_frame[:, w:, :] = right_frame

                writer.write(merged_frame)


if __name__ == '__main__':
    configure_logging(logging.INFO)
    main(parse_args())
