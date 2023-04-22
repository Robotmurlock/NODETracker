"""
Evaluate Kalman Filter
"""
import argparse
import functools
import json
import logging
import os
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterable

import numpy as np
import torch
import yaml
from tqdm import tqdm

from nodetracker.common.project import ASSETS_PATH, OUTPUTS_PATH
from nodetracker.datasets import dataset_factory, TrajectoryDataset
from nodetracker.library.cv.bbox import BBox
from nodetracker.library.kalman_filter.botsort_kf import BotSortKalmanFilter
from nodetracker.utils.logging import configure_logging

logger = logging.getLogger('KF_E2E_EVAL')


Vector = Union[List[float], np.ndarray, torch.Tensor]


def parse_args() -> argparse.Namespace:
    """
    Parses script arguments.

    Returns:
        Parse script arguments
    """
    parser = argparse.ArgumentParser(description='Kalman Filter evaluation')
    parser.add_argument('--input-path', type=str, required=False, default=ASSETS_PATH, help='Datasets path.')
    parser.add_argument('--output-path', type=str, required=False, default=OUTPUTS_PATH, help='Output path.')
    parser.add_argument('--split-index-name', type=str, required=False, default='.split_index_3to1to4.yaml', help='Split index name.')
    parser.add_argument('--split', type=str, required=False, default='val', help='Dataset split name.')
    parser.add_argument('--dataset-name', type=str, required=True, help='Dataset name.')
    parser.add_argument('--steps', type=int, required=False, default=1, help='Number of forecast steps.')
    parser.add_argument('--optimal-kf', action='store_true', help='Use optimal KF motion matrix.')
    parser.add_argument('--noise-sigma', type=float, required=False, default=0.05,
                        help='Add noise for posterior evaluation. Set 0 to disable.')
    parser.add_argument('--skip-det-proba', type=float, required=False, default=0.0, help='Probability to skip detection.')
    parser.add_argument('--n-workers', type=int, required=False, default=8, help='Number of workers for faster evaluation.')
    parser.add_argument('--dataset-type', type=str, required=False, default='MOT20', help='Supported: MOT20, LaSOT')
    return parser.parse_args()


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


def merge_metrics(global_metrics: Optional[Dict[str, List[float]]], metrics: Optional[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    """
    Adds aggregated metrics to global metrics dictionary.
    If global dictionary is empty then aggregated dictionary is taken as the global one.
    if added metrics are None then global metrics is returned instead.

    Args:
        global_metrics: Global metrics data
        metrics: Aggregated metrics data (can be empty)

    Returns:
        Merged (global) metrics
    """
    if metrics is None:
        return global_metrics

    if global_metrics is None:
        return metrics

    metric_names = set(global_metrics.keys())
    assert metric_names == set(metrics.keys()), f'Metric keys do not match: {metric_names} != {set(metrics.keys())}'

    for name in metric_names:
        global_metrics[name].extend(metrics[name])

    return global_metrics


def simulate_detector_noise(
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


def calc_iou(pred: Union[List[float], np.ndarray, torch.Tensor], gt: Union[List[float], np.ndarray, torch.Tensor]) -> float:
    """
    Calculates iou score between two bboxes represented as numpy arrays in yxwh format.

    Args:
        pred: Prediction bbox (numpy)
        gt: Ground Truth bbox (numpy)

    Returns:
        IOU score
    """
    pred_bbox = BBox.from_yxwh(*pred, clip=True)
    gt_bbox = BBox.from_yxwh(*gt, clip=True)
    return float(gt_bbox.iou(pred_bbox))


def simulate_detector_false_positive(proba: float, first_frame: bool) -> bool:
    """
    "Deletes" measurement with some probability. First frame can't be skipped.

    Args:
        proba: Probability to skip detection
        first_frame: Is it first frame for the object

    Returns:
        True if detection is skipped else False
    """
    if first_frame:
        return False

    r = random.uniform(0, 1)
    return r < proba


def kf_trak_eval(
    measurements: List[List[float]],
    det_noise_sigma: float,
    det_skip_proba: float,
    n_pred_steps: int,
    use_optimal_motion_mat: bool = False
) -> Optional[Dict[str, List[float]]]:
    """
    Evaluates Kalman filter on SOT tracking.

    Args:

        measurements: List of measurements
        det_noise_sigma: Detection noise (sigma)
        det_skip_proba: Detection skip probability
        n_pred_steps: Number of forward inference steps
        use_optimal_motion_mat: Use optimal (fine-tuned) KF

    Returns:
        Metrics for each tracker step.
    """
    kf = BotSortKalmanFilter(use_optimal_motion_mat=use_optimal_motion_mat)  # Create KF for new track
    mean, covariance, mean_hat, covariance_hat, prev_measurement = None, None, None, None, None
    sample_metrics = defaultdict(list)

    n_data_points = len(measurements)
    n_inf_steps = n_data_points - n_pred_steps
    if n_inf_steps <= 0:
        return None

    for index in range(n_inf_steps):
        measurement = measurements[index]
        measurement_no_noise = measurement
        measurement = simulate_detector_noise(measurement, prev_measurement, det_noise_sigma)
        skip_detection = simulate_detector_false_positive(det_skip_proba, first_frame=prev_measurement is None)

        if mean is None:
            # First KF iteration
            mean, covariance = kf.initiate(measurement)
            prior, _ = kf.project(mean, covariance)
            # Nothing to compare in first iteration
        else:
            # Perform prediction
            mean_hat, covariance_hat = kf.predict(mean, covariance)  # Prior
            if skip_detection:
                mean, covariance = mean_hat, covariance_hat
            else:
                mean, covariance = kf.update(mean_hat, covariance_hat, measurement)  # Posterior

            total_mse = 0.0
            total_iou = 0.0
            forward_mean_hat, forward_covariance_hat = mean_hat, covariance_hat
            for p_index in range(n_pred_steps):
                prior, _ = kf.project(forward_mean_hat, forward_covariance_hat)  # KF Prediction is before update

                gt = np.array(measurements[index + p_index], dtype=np.float32)

                # Calculate MSE
                mse = ((gt - prior) ** 2).mean()
                sample_metrics[f'prior-MSE-{p_index}'].append(mse)
                total_mse += mse

                # Calculate IOU (Accuracy)
                iou_score = calc_iou(prior, gt)
                sample_metrics[f'prior-Accuracy-{p_index}'].append(iou_score)
                total_iou += iou_score

                forward_mean_hat, forward_covariance_hat = kf.predict(forward_mean_hat, forward_covariance_hat)

            sample_metrics['prior-MSE'].append(total_mse / n_pred_steps)
            sample_metrics['prior-Accuracy'].append(total_iou / n_pred_steps)

            posterior, _ = kf.project(mean, covariance)
            gt = np.array(measurement_no_noise, dtype=np.float32)

            posterior_mse = ((posterior - gt) ** 2).mean()
            sample_metrics[f'posterior-MSE'].append(posterior_mse)
            posterior_iou_score = calc_iou(posterior, gt)
            sample_metrics[f'posterior-Accuracy'].append(posterior_iou_score)

            # Save data
            prev_measurement = measurement

    return sample_metrics


def create_object_id_iterator(dataset: TrajectoryDataset, scene_name: str) -> Iterable[List[List[float]]]:
    object_ids = dataset.get_scene_object_ids(scene_name)
    for object_id in object_ids:
        n_data_points = dataset.get_object_data_length(object_id)
        measurements = [dataset.get_object_data_label(object_id, index)['bbox']
                        for index in range(n_data_points)]
        yield measurements


def main(args: argparse.Namespace) -> None:
    dataset_path = os.path.join(args.input_path, args.dataset_name)
    split_index_path = os.path.join(args.input_path, args.dataset_name, args.split_index_name)
    with open(split_index_path, 'r', encoding='utf-8') as f:
        split_index = yaml.safe_load(f)

    # Parameters
    n_pred_steps: int = args.steps
    det_skip_proba: float = args.skip_det_proba
    det_noise_sigma: float = args.noise_sigma
    n_workers: int = args.n_workers

    logger.info(f'Loading dataset from path "{dataset_path}"')
    dataset = dataset_factory(
        name=args.dataset_type,
        path=dataset_path,
        sequence_list=split_index[args.split],
        history_len=1,  # Not relevant
        future_len=1  # not relevant
    )

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    metrics_path = os.path.join(args.output_path, f'kf_metrics_noise{det_noise_sigma}_skip{det_skip_proba}.json')
    logger.info('Evaluating Kalman Filter')

    global_metrics = None
    scene_names = dataset.scenes
    for scene_name in scene_names:
        object_id_iterator = create_object_id_iterator(dataset, scene_name)
        n_object_ids = dataset.get_scene_number_of_object_ids(scene_name)
        with Pool(n_workers) as pool:
            method = functools.partial(
                kf_trak_eval,
                det_skip_proba=det_skip_proba,
                det_noise_sigma=det_noise_sigma,
                n_pred_steps=n_pred_steps,
                use_optimal_motion_mat=args.optimal_kf
            )

            for sample_metrics in tqdm(pool.imap_unordered(method, object_id_iterator),
                                       unit='track', desc=f'Evaluating tracker on {scene_name}', total=n_object_ids):
                global_metrics = merge_metrics(global_metrics, sample_metrics)

    metrics = aggregate_metrics(global_metrics)

    # Save metrics
    logger.info(f'Metrics:\n{json.dumps(metrics, indent=2)}')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    configure_logging(logging.DEBUG)
    main(parse_args())
