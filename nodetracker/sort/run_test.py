"""
Script for SortTracker testing
"""
import argparse
import os
from pathlib import Path
from tqdm import tqdm

import cv2
import pandas as pd

from nodetracker.library.cv import PredBBox, BBox


def run_test(args: argparse.Namespace) -> None:
    output_path: str = args.output_path
    df = pd.read_csv(args.dataset_path)
    df = df.sort_values(by='frame_id')
    groups = df.groupby('frame_id')

    mp4_path = os.path.join(output_path, 'gt.mp4')
    Path(output_path).mkdir(parents=True, exist_ok=True)
    resolution = (1280, 720)

    # noinspection PyUnresolvedReferences
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # noinspection PyUnresolvedReferences
    mp4_writer = cv2.VideoWriter(mp4_path, fourcc, 10, resolution)

    for frame_id, df_grp in tqdm(groups, desc='Visualizing', unit='frame'):
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


def parse_args() -> argparse.Namespace:
    """
    Parses script arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='SortTracker testing.')
    parser.add_argument('--dataset-path', type=str, required=True, help='Tracker sample dataset path')
    parser.add_argument('--output-path', type=str, required=True, help='Output dir path')
    return parser.parse_args()


if __name__ == '__main__':
    run_test(parse_args())
