"""
Script for SortTracker testing
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Optional
import yaml

import cv2
import pandas as pd
from tqdm import tqdm

from nodetracker.library.cv import PredBBox, BBox, MP4Writer
from nodetracker.node.factory import load_or_create_model
from nodetracker.datasets import transforms
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
    with MP4Writer(mp4_path, fps=15, shape=(1920, 1080), resize=True) as writer:
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

            writer.write(image)


def create_tracker_mp4(
    df: pd.DataFrame,
    mp4_path: str,
    model_type: str,
    model_params: dict,
    transform_type: str,
    transform_params: dict,
    checkpoint_path: Optional[str] = None,
    accelerator: str = 'cpu'
):
    """
    Create tracker output

    Args:
        df: data
        mp4_path: output path
        model_type: forecaster model
        model_params: forecaster parameters
        transform_type: forecaster data transform type
        transform_params: forecaster data transform params
        checkpoint_path: forecaster checkpoint path
        accelerator: forecaster accelerator (cpu/gpu)
    """
    forecaster = load_or_create_model(
        model_type=model_type,
        params=model_params,
        checkpoint_path=checkpoint_path
    )
    tensor_transform = transforms.transform_factory(
        name=transform_type,
        params=transform_params
    )
    matcher = HungarianAlgorithmIOU(match_threshold=0.3)
    tracker = SortTracker(
        forecaster=forecaster,
        matcher=matcher,
        tensor_transform=tensor_transform,
        accelerator=accelerator
    )
    groups = df.groupby('frame_id')

    Path(mp4_path).parent.mkdir(parents=True, exist_ok=True)
    with MP4Writer(mp4_path, fps=15, shape=(1920, 1080), resize=True) as writer:
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

            writer.write(image)


def run_test(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.dataset_path)
    df = df.sort_values(by='frame_id')

    model_params = {}
    if args.model_params is not None:
        with open(args.model_params, 'r', encoding='utf-8') as f:
            model_params = yaml.safe_load(f)

    transform_params = {}
    if args.transform_params is not None:
        with open(args.transform_params, 'r', encoding='utf-8') as f:
            transform_params = yaml.safe_load(f)

    create_tracker_mp4(
        df=df,
        mp4_path=os.path.join(args.output_path, 'tracker.mp4'),
        model_type=args.model_type,
        model_params=model_params,
        checkpoint_path=args.model_checkpoint,
        transform_type=args.transform_type,
        transform_params=transform_params,
        accelerator=args.accelerator
    )
    if args.generate_gt:
        create_gt_mp4(
            df=df,
            mp4_path=os.path.join(args.output_path, 'gt.mp4')
        )


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

    parser.add_argument('--model-type', type=str, required=True, help='Model (forecaster) type')
    parser.add_argument('--model-params', type=str, required=False, default=None, help='Model (forecaster) params')
    parser.add_argument('--model-checkpoint', type=str, required=False, default=None, help='Model (forecaster) checkpoint path')

    parser.add_argument('--transform-type', type=str, required=False, default='identity', help='Forecaster input data transform type')
    parser.add_argument('--transform-params', type=str, required=False, default=None, help='Forecaster input data transform params')

    parser.add_argument('--accelerator', type=str, required=False, default='cpu', help='Forecaster accelerator (cpu/gpu)')
    return parser.parse_args()


if __name__ == '__main__':
    configure_logging(logging.DEBUG)
    run_test(parse_args())
