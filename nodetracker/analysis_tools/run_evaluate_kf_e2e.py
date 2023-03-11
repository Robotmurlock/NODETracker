"""
Evaluate Kalman Filter
"""
import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from nodetracker.common.project import ASSETS_PATH, OUTPUTS_PATH
from nodetracker.datasets.mot.core import MOTDataset, LabelType
from nodetracker.library.cv.bbox import BBox
from nodetracker.library.kalman_filter.botsort_kf import BotSortKalmanFilter
from nodetracker.utils.logging import configure_logging

logger = logging.getLogger('MotTools')


def aggregate_metrics(sample_metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Aggregates dictionary of sample metrics (list of values) to dictionary of metrics.
    All metrics should be "mean-friendly".

    Args:
        sample_metrics: Sample metrics

    Returns:
        Aggregated metrics.
    """
    return {k: sum(v) / len(v) for k, v in sample_metrics.items()}


def parse_args() -> argparse.Namespace:
    """
    Returns:
        Parse tool arguments
    """
    parser = argparse.ArgumentParser(description='Kalman Filter evaluation')
    parser.add_argument('--input-path', type=str, required=False, default=ASSETS_PATH, help='Datasets path.')
    parser.add_argument('--output-path', type=str, required=False, default=OUTPUTS_PATH, help='Output path.')
    parser.add_argument('--split', type=str, required=False, default='val', help='Dataset split name.')
    parser.add_argument('--dataset-name', type=str, required=True, help='Dataset name.')
    parser.add_argument('--steps', type=int, required=False, default=1, help='Number of forecast steps.')
    parser.add_argument('--noise-sigma', type=float, required=False, default=0.05,
                        help='Add noise for posterior evaluation. Set 0 to disable.')
    return parser.parse_args()


def add_measurement_noise(
    measurement: List[float],
    prev_measurement: Optional[List[float]],
    sigma: float
) -> List[float]:
    """
    Adds measurements noise (random variable) to measurement. Noise covariance matrix for step k is a diagonal matrix with values:
    [ (sigma * h[k-1])^2, (sigma * w[k-1])^2, (sigma * h[k-1])^2, (sigma * w[k-1])^2 ]

    Noise is time-dependant on previous bbox width and height.

    Args:
        measurement: Current measurement (ground truth)
        prev_measurement: Previous measurement
        sigma: Noise multiplier

    Returns:
        Measurement with added noise
    """
    if prev_measurement is None:
        return measurement  # First measurement does not have any noise
    x, y, h, w = measurement
    _, _, p_h, p_w = prev_measurement
    x_noise, h_noise = sigma * np.random.randn(2) * p_h
    y_noise, w_noise = sigma * np.random.randn(2) * p_w

    return [
        x + x_noise,
        y + y_noise,
        h + h_noise,
        w + w_noise
    ]


def main(args: argparse.Namespace) -> None:
    dataset_path = os.path.join(args.input_path, args.dataset_name, args.split)
    n_pred_steps = args.steps
    logger.info(f'Loading dataset from path "{dataset_path}"')
    dataset = MOTDataset(
        path=dataset_path,
        history_len=1,  # Not relevant
        future_len=1,  # not relevant
        label_type=LabelType.GROUND_TRUTH
    )

    sample_metrics = defaultdict(list)

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    metrics_path = os.path.join(args.output_path, 'kf_metrics.json')
    logger.info('Evaluating Kalman Filter')

    scene_names = dataset.scenes
    for scene_name in scene_names:
        object_ids = dataset.get_scene_object_ids(scene_name)
        for object_id in tqdm(object_ids, unit='track', desc=f'Evaluating tracker on {scene_name}'):
            kf = BotSortKalmanFilter()  # Create KF for new track
            mean, covariance, mean_hat, covariance_hat, prev_measurement = None, None, None, None, None

            n_data_points = dataset.get_object_data_length(object_id)
            for index in range(n_data_points-n_pred_steps):
                measurement = dataset.get_object_data_label(object_id, index)['bbox']
                measurement = add_measurement_noise(measurement, prev_measurement, args.noise_sigma)

                if mean is None:
                    # First KF iteration
                    mean, covariance = kf.initiate(measurement)
                    prior, _ = kf.project(mean, covariance)
                    # Nothing to compare in first iteration
                else:
                    # Perform prediction
                    mean_hat, covariance_hat = kf.predict(mean, covariance)
                    mean, covariance = kf.update(mean_hat, covariance_hat, measurement)

                    total_mse = 0.0
                    total_iou = 0.0
                    for p_index in range(1, n_pred_steps+1):
                        prior, _ = kf.project(mean_hat, covariance_hat)  # KF Prediction is before update

                        gt = np.array(dataset.get_object_data_label(object_id, index + p_index)['bbox'], dtype=np.float32)

                        # Calculate MSE
                        mse = ((gt - prior) ** 2).mean()
                        sample_metrics[f'MSE-{p_index}'].append(mse)
                        total_mse += mse

                        # Calculate IOU (Accuracy)
                        gt_bbox = BBox.from_yxwh(*gt, clip=True)
                        pred_bbox = BBox.from_yxwh(*prior, clip=True)
                        iou_score = gt_bbox.iou(pred_bbox)
                        sample_metrics[f'posterior-Accuracy-{p_index}'].append(iou_score)
                        total_iou += iou_score

                        mean_hat, covariance_hat = kf.predict(mean_hat, covariance_hat)

                    sample_metrics['MSE'].append(total_mse / n_pred_steps)
                    sample_metrics['Accuracy'].append(total_iou / n_pred_steps)

                    # Posterior eval
                    # Calculate MSE
                    posterior, _ = kf.project(mean, covariance)
                    gt = np.array(measurement, dtype=np.float32)

                    posterior_mse = ((posterior - gt) ** 2).mean()
                    sample_metrics[f'posterior-MSE'].append(posterior_mse)

                    # Calculate IOU (Accuracy)
                    posterior_bbox = BBox.from_yxwh(*posterior, clip=True)
                    gt_bbox = BBox.from_yxwh(*gt, clip=True)
                    posterior_iou_score = gt_bbox.iou(posterior_bbox)
                    sample_metrics[f'posterior-Accuracy'].append(posterior_iou_score)

                    # Save data
                    prev_measurement = measurement

    metrics = aggregate_metrics(sample_metrics)

    # Save metrics
    logger.info(f'Metrics: \n{json.dumps(metrics, indent=2)}')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    configure_logging(logging.DEBUG)
    main(parse_args())
