"""
Object Movement analysis.
"""
import argparse
import os
from pathlib import Path
from typing import List, Any

import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from nodetracker.common.project import ASSETS_PATH, OUTPUTS_PATH
from nodetracker.datasets import dataset_factory, TrajectoryDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Movement Analysis')
    parser.add_argument('--input-path', type=str, default=ASSETS_PATH, required=False, help='Path where datasets are stored.')
    parser.add_argument('--output-path', type=str, default=os.path.join(OUTPUTS_PATH, 'movement_analysis'), required=False,
                        help='Path where the script output is stored.')
    parser.add_argument('--dataset-name', type=str, required=True, help='Dataset name in datasets (input) path.')
    parser.add_argument('--dataset-type', type=str, required=True, help='MOT or LaSOT.')
    return parser.parse_args()


def extract_data(dataset: TrajectoryDataset) -> List[List[Any]]:
    """
    Extracts difference data from the dataset.

    Args:
        dataset: Loaded dataset

    Returns:
        Difference data (object_id, index, category, abs bbox difference)
    """
    data = []
    for scene in tqdm(dataset.scenes, unit='scene', desc='Extracting data'):
        object_ids = dataset.get_scene_object_ids(scene)
        for object_id in object_ids:
            category = dataset.get_object_category(object_id)
            length = dataset.get_object_data_length(object_id)

            for index in range(length - 1):
                prev_data = dataset.get_object_data_label(object_id, index)
                next_data = dataset.get_object_data_label(object_id, index + 1)
                if prev_data['occ'] or prev_data['oov'] or next_data['occ'] or next_data['occ']:
                    # Skip occluded and out of view bboxes
                    continue

                prev_bbox = np.array(prev_data['bbox'], dtype=np.float32)
                next_bbox = np.array(next_data['bbox'], dtype=np.float32)
                bbox_diff = float(np.abs(next_bbox - prev_bbox).sum())
                data.append([object_id, index, category, bbox_diff])

    return data


def perform_analysis(data: List[List[Any]], output_path: str) -> None:
    """
    Creates plots based on extracted dataset data.

    Args:
        data: Extracted data
        output_path: Path where to store plots
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data, columns=['object_id', 'index', 'category', 'diff'])

    # Configuration
    n_bins = 50
    hist_type = 'probability density'

    # Plot global histogram
    fig = px.histogram(df, x='diff', nbins=n_bins, histnorm=hist_type)
    fig.write_image(os.path.join(output_path, 'global_histogram.png'))

    category_groups = df.groupby('category')
    for category, df_category in category_groups:
        fig = px.histogram(df_category, x='diff', nbins=n_bins, histnorm=hist_type)
        fig.write_image(os.path.join(output_path, f'{category}_histogram.png'))


def main(args: argparse.Namespace) -> None:
    additional_params = {'skip_corrupted': True} if args.dataset_name == 'MOT20' else None

    dataset = dataset_factory(
        name=args.dataset_name,
        path=os.path.join(args.input_path, args.dataset_name),
        history_len=1,  # Not relevant
        future_len=1,  # Not relevant
        additional_params=additional_params
    )

    data = extract_data(dataset)
    perform_analysis(data, os.path.join(args.output_path, args.dataset_name))


if __name__ == '__main__':
    main(parse_args())
