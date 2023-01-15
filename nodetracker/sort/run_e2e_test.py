"""
Script for SortTracker testing
"""
import argparse
import logging
import os
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from nodetracker.library.cv import PredBBox, BBox
from nodetracker.node.kalman_filter import TorchConstantVelocityODKalmanFilter
from nodetracker.sort import SortTracker, HungarianAlgorithmIOU
from nodetracker.utils.logging import configure_logging


def create_gt_mp4(df: pd.DataFrame, mp4_path: str) -> None:
    """
    Create ground truth video.

    Args:
        df: data
        mp4_path: video output path
    """
    groups = df.groupby('frame_id')

    Path(mp4_path).parent.mkdir(parents=True, exist_ok=True)
    resolution = (1920, 1080)

    # noinspection PyUnresolvedReferences
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # noinspection PyUnresolvedReferences
    mp4_writer = cv2.VideoWriter(mp4_path, fourcc, 10, resolution)

    for _, df_grp in tqdm(groups, desc='Visualizing', unit='frame'):
        image_path = df_grp.iloc[0]['image_path']

        # noinspection PyUnresolvedReferences
        image = cv2.imread(image_path)
        assert image is not None, 'Opening image failed!'

        for _, row in df_grp.iterrows():
            object_id, xmin, ymin, h, w = row[['object_id', 'xmin', 'ymin', 'h', 'w']]
            conf = 1.0
            bbox = PredBBox.create(BBox.from_xyhw(xmin, ymin, h, w, clip=True), label=object_id, conf=conf)
            image = bbox.draw(image)

        # noinspection PyUnresolvedReferences
        image = cv2.resize(image, resolution)
        mp4_writer.write(image)

    mp4_writer.release()


def create_tracker_mp4(df: pd.DataFrame, mp4_path: str):
    """
    Create tracker output

    Args:
        df: data
        mp4_path: output path
    """
    forecaster = TorchConstantVelocityODKalmanFilter(
        time_step_multiplier=3e-2,
        process_noise_multiplier=1,
        measurement_noise_multiplier=1
    )
    matcher = HungarianAlgorithmIOU(match_threshold=0.3)
    tracker = SortTracker(
        forecaster=forecaster,
        matcher=matcher
    )
    groups = df.groupby('frame_id')

    Path(mp4_path).parent.mkdir(parents=True, exist_ok=True)
    resolution = (1920, 1080)

    # noinspection PyUnresolvedReferences
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # noinspection PyUnresolvedReferences
    mp4_writer = cv2.VideoWriter(mp4_path, fourcc, 10, resolution)

    for _, df_grp in tqdm(groups, desc='Visualizing', unit='frame'):
        image_path = df_grp.iloc[0]['image_path']

        # noinspection PyUnresolvedReferences
        image = cv2.imread(image_path)
        assert image is not None, 'Opening image failed!'

        bboxes = []
        for _, row in df_grp.iterrows():
            object_id, xmin, ymin, h, w = row[['object_id', 'xmin', 'ymin', 'h', 'w']]
            conf = 1.0
            bbox = PredBBox.create(BBox.from_xyhw(xmin, ymin, h, w, clip=True), label=object_id, conf=conf)
            bboxes.append(bbox)

        tracklet_bboxes = tracker.track(bboxes)
        for bbox in tracklet_bboxes:
            bbox.draw(image)

        # noinspection PyUnresolvedReferences
        image = cv2.resize(image, resolution)
        mp4_writer.write(image)

    mp4_writer.release()


def run_test(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.dataset_path)
    df = df.sort_values(by='frame_id')
    create_tracker_mp4(df, os.path.join(args.output_path, 'tracker.mp4'))
    if args.generate_gt:
        create_gt_mp4(df, os.path.join(args.output_path, 'gt.mp4'))


def parse_args() -> argparse.Namespace:
    """
    Parses script arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='SortTracker testing.')
    parser.add_argument('--dataset-path', type=str, required=True, help='Tracker sample dataset path')
    parser.add_argument('--output-path', type=str, required=True, help='Output dir path')
    parser.add_argument('--generate-gt', type=bool, required=False, default=True, help='Generate gt.mp4 file')
    return parser.parse_args()


if __name__ == '__main__':
    configure_logging(logging.DEBUG)
    run_test(parse_args())
