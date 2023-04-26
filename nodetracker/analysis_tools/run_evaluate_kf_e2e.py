"""
Evaluate Kalman Filter
"""
import argparse
import functools
import json
import logging
import os
import random
import traceback
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterable, Tuple, Any

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from nodetracker.common.project import ASSETS_PATH, OUTPUTS_PATH
from nodetracker.datasets import dataset_factory, TrajectoryDataset
from nodetracker.evaluation import sot as sot_metrics
from nodetracker.library.cv.bbox import BBox
from nodetracker.library.cv import color_palette
from nodetracker.library.cv.video_writer import MP4Writer
from nodetracker.library.cv import drawing
from nodetracker.library.kalman_filter.botsort_kf import BotSortKalmanFilter
from nodetracker.utils.logging import configure_logging

METRICS = [
    ('MSE', sot_metrics.mse),
    ('Accuracy', sot_metrics.accuracy),
    ('Success', sot_metrics.success),
    ('NormPrecision', sot_metrics.norm_precision)
]


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
    parser.add_argument('--skip-occ', action='store_true', help='Skip occlusions during inference.')
    parser.add_argument('--skip-oov', action='store_true', help='Skip out of view during inference.')
    parser.add_argument('--noise-sigma', type=float, required=False, default=0.05,
                        help='Add noise for posterior evaluation. Set 0 to disable.')
    parser.add_argument('--skip-det-proba', type=float, required=False, default=0.0, help='Probability to skip detection.')
    parser.add_argument('--n-workers', type=int, required=False, default=8, help='Number of workers for faster evaluation.')
    parser.add_argument('--dataset-type', type=str, required=False, default='MOT20', help='Supported: MOT20, LaSOT')
    parser.add_argument('--visualize', action='store_true', required=False, help='Create visualization video.')
    parser.add_argument('--video-dirpath', type=str, default='KF_visualization', required=False,
                        help='Visualization dirpath relative to the `output-path`.')
    parser.add_argument('--visualize-show-iou', action='store_true', required=False, help='Draw IOU (accuracy) one images.')
    return parser.parse_args()


def aggregate_metrics(sample_metrics: Any) -> Any:
    """
    Aggregates dictionary of sample metrics (list of values) to dictionary of metrics.
    All metrics should be "mean-friendly".

    Args:
        sample_metrics: Sample metrics

    Returns:
        Aggregated metrics.
    """
    if isinstance(sample_metrics, list):
        return sum(sample_metrics) / len(sample_metrics)
    elif isinstance(sample_metrics, dict):
        agg = {k: aggregate_metrics(v) for k, v in sample_metrics.items()}
        return dict(sorted(agg.items()))
    else:
        return sample_metrics


def merge_metrics(
    global_metrics: Optional[Dict[str, Any]],
    metrics: Optional[Dict[str, List[float]]],
    scene_name: str,
    category: str
) -> Dict[str, Any]:
    """
    Adds aggregated metrics to global metrics dictionary.
    If global dictionary is empty then aggregated dictionary is taken as the global one.
    if added metrics are None then global metrics is returned instead.

    Args:
        global_metrics: Global metrics data
        metrics: Aggregated metrics data (can be empty)
        scene_name: Save scene name metrics
        category: Category

    Returns:
        Merged (global) metrics
    """
    if metrics is None:
        return global_metrics

    if global_metrics is None:
        return {
            'global': metrics,
            'categories': {
                category: metrics
            },
            'scenes': {
                scene_name: metrics
            }
        }

    metric_names = set(global_metrics['global'].keys())
    assert metric_names == set(metrics.keys()), f'Metric keys do not match: {metric_names} != {set(metrics.keys())}'

    for name in metric_names:
        global_metrics['global'][name].extend(metrics[name])
        if category not in global_metrics['categories']:
            global_metrics['categories'][category] = {}
        if name not in global_metrics['categories'][category]:
            global_metrics['categories'][category][name] = []
        global_metrics['categories'][category][name].extend(metrics[name])

    assert scene_name not in global_metrics['scenes'], f'Found duplicate scene "{scene_name}"!'
    global_metrics['scenes'][scene_name] = metrics

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
    object_id_traj_data: Tuple[str, List[dict]],
    det_noise_sigma: float,
    det_skip_proba: float,
    n_pred_steps: int,
    use_optimal_motion_mat: bool = False,
    skip_occ: bool = False,
    skip_oov: bool = False,
    fast_inference: bool = True
) -> Optional[Tuple[str, Dict[str, List[float]], Dict[str, List[List[float]]]]]:
    """
    Evaluates Kalman filter on SOT tracking.

    Args:

        object_id_traj_data: (Object id, Trajectory point data)
        det_noise_sigma: Detection noise (sigma)
        det_skip_proba: Detection skip probability
        n_pred_steps: Number of forward inference steps
        use_optimal_motion_mat: Use optimal (fine-tuned) KF
        fast_inference: Do not save inference data (optimization)
        skip_occ: Skip occlusions during inference
        skip_oov: Skip out of view during inference

    Returns:
        Metrics for each tracker step.
    """
    object_id, traj_data = object_id_traj_data

    measurements = [ti['bbox'] for ti in traj_data]
    frame_ids = [ti['frame_id'] for ti in traj_data]
    occlusions = [ti['occ'] for ti in traj_data]
    out_of_view = [ti['oov'] for ti in traj_data]

    kf = BotSortKalmanFilter(use_optimal_motion_mat=use_optimal_motion_mat)  # Create KF for new track
    mean, covariance, mean_hat, covariance_hat, prev_measurement = None, None, None, None, None
    sample_metrics: Dict = defaultdict(list)
    inference = {k: {} for k in ['prior', 'posterior']}

    n_data_points = len(measurements)
    n_inf_steps = n_data_points - n_pred_steps
    if n_inf_steps <= 0:
        return None

    for index in range(n_inf_steps):
        measurement = measurements[index]
        frame_id = frame_ids[index]
        measurement_no_noise = measurement
        measurement = simulate_detector_noise(measurement, prev_measurement, det_noise_sigma)
        skip_detection = simulate_detector_false_positive(det_skip_proba, first_frame=prev_measurement is None)

        if skip_oov:
            oov = out_of_view[index]
            if oov:
                mean, covariance, mean_hat, covariance_hat, prev_measurement = None, None, None, None, None
                continue

        if skip_occ:
            occ = occlusions[index]
            if occ:
                mean, covariance, mean_hat, covariance_hat, prev_measurement = None, None, None, None, None
                continue

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
                # noinspection PyBroadException
                try:
                    mean, covariance = kf.update(mean_hat, covariance_hat, measurement)  # Posterior
                except:
                    logger.error(f'Error occurred on update: {traceback.format_exc()}!')
                    mean, covariance = mean_hat, covariance_hat

            total_score = {k: 0.0 for k, _ in METRICS}

            forward_mean_hat, forward_covariance_hat = mean_hat, covariance_hat
            first_prior = None
            for p_index in range(n_pred_steps):
                prior, _ = kf.project(forward_mean_hat, forward_covariance_hat)  # KF Prediction is before update
                first_prior = prior if first_prior is None else first_prior
                gt = np.array(measurements[index + p_index], dtype=np.float32)

                for metric_name, metric_func in METRICS:
                    score = metric_func(gt, prior)
                    sample_metrics[f'prior-{metric_name}-{p_index}'].append(score)
                    total_score[metric_name] += score

                forward_mean_hat, forward_covariance_hat = kf.predict(forward_mean_hat, forward_covariance_hat)

            for metric_name, _ in METRICS:
                sample_metrics[f'prior-{metric_name}'].append(total_score[metric_name] / n_pred_steps)

            posterior, _ = kf.project(mean, covariance)
            gt = np.array(measurement_no_noise, dtype=np.float32)

            for metric_name, metric_func in METRICS:
                score = metric_func(posterior, gt)
                sample_metrics[f'posterior-{metric_name}'].append(score)

            # Save data
            prev_measurement = measurement

            if not fast_inference:
                inference['prior'][frame_id] = first_prior.tolist()
                inference['posterior'][frame_id] = posterior.tolist()

    return object_id, dict(sample_metrics), inference


def create_object_id_iterator(dataset: TrajectoryDataset, scene_name: str) \
        -> Iterable[Tuple[str, List[dict]]]:
    """
    Creates object id iterator. Creates input data for motion model for each object existing in the scene.

    Args:
        dataset: Dataset
        scene_name: Scene (sequence) name

    Returns:
        object_id, Trajectory data
    """
    object_ids = dataset.get_scene_object_ids(scene_name)
    for object_id in object_ids:
        n_data_points = dataset.get_object_data_length(object_id)
        traj_data = [dataset.get_object_data_label(object_id, index) for index in range(n_data_points)]
        yield object_id, traj_data


def visualize_video(
    scene_name: str,
    inference_visualization_cache: Dict[str, Dict[str, Dict[int, List[float]]]],
    dataset: TrajectoryDataset,
    output_path: str,
    video_dirpath: str,
    model_name: str,
    show_iou: bool = False
) -> None:
    inference_types = list(next(iter(inference_visualization_cache.values())).keys())
    for inference_type in inference_types:

        full_video_dirpath = os.path.join(output_path, video_dirpath, inference_type)
        Path(full_video_dirpath).mkdir(parents=True, exist_ok=True)
        scene_info = dataset.get_scene_info(scene_name)
        n_frames = scene_info.seqlength  # TODO: Works for LaSOT and MOT but may crash for other datasets

        video_path = os.path.join(full_video_dirpath, f'{scene_name}.mp4')

        logger.info(f'Performing visualization for scene "{scene_name}" and inference type "{inference_type}"... '
                    f'Video path is "{video_path}"')
        with MP4Writer(video_path, fps=30) as writer:
            for frame_index in tqdm(range(n_frames), desc='Drawing', unit='frame', total=n_frames):
                image_path = dataset.get_scene_image_path(scene_name, frame_index)
                # noinspection PyUnresolvedReferences
                image = cv2.imread(image_path)

                for object_id, inference_data in inference_visualization_cache.items():
                    gt_data = dataset.get_object_data_label_by_frame_index(object_id, frame_index)
                    if gt_data is not None:
                        gt_bbox = BBox.from_yxwh(*gt_data['bbox'], clip=True)
                        image = gt_bbox.draw(image, color=color_palette.GREEN)

                    pred_data = inference_data[inference_type].get(frame_index)
                    if pred_data is not None:
                        pred_bbox = BBox.from_yxwh(*pred_data, clip=True)
                        image = pred_bbox.draw(image, color=color_palette.RED)

                        if show_iou and gt_data is not None:
                            iou_score = gt_bbox.iou(pred_bbox)
                            left, top, _, _ = pred_bbox.scaled_yxyx_from_image(image)
                            image = drawing.draw_text(image, f'{100*iou_score:.1f}', left, top - 3, color_palette.RED)

                image = drawing.draw_text(image, 'GroundTruth', 5, 15, color_palette.GREEN)
                image = drawing.draw_text(image, f'{model_name}-{inference_type}', 5, 30, color_palette.RED)

                writer.write(image)


def main(args: argparse.Namespace) -> None:
    dataset_path = os.path.join(args.input_path, args.dataset_name)
    split_index_path = os.path.join(args.input_path, args.dataset_name, args.split_index_name)
    with open(split_index_path, 'r', encoding='utf-8') as f:
        split_index = yaml.safe_load(f)
    skip_occ: bool = args.skip_occ
    skip_oov: bool = args.skip_oov

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
        scene_info = dataset.get_scene_info(scene_name)
        inference_visualization_cache: Dict[str, Dict[str, Dict[int, List[float]]]] = {}  # Used only for visualization

        with Pool(n_workers) as pool:
            method = functools.partial(
                kf_trak_eval,
                det_skip_proba=det_skip_proba,
                det_noise_sigma=det_noise_sigma,
                n_pred_steps=n_pred_steps,
                use_optimal_motion_mat=args.optimal_kf,
                skip_occ=skip_occ,
                skip_oov=skip_oov,
                fast_inference=not args.visualize
            )

            for inference_package in tqdm(pool.imap_unordered(method, object_id_iterator),
                                       unit='track', desc=f'Evaluating tracker on {scene_name}', total=n_object_ids):
                if inference_package is None:
                    continue

                object_id, scene_metrics, inference = inference_package

                global_metrics = merge_metrics(
                    global_metrics=global_metrics,
                    metrics=scene_metrics,
                    scene_name=scene_name,
                    category=scene_info.category
                )

                if args.visualize:
                    inference_visualization_cache[object_id] = inference

        if args.visualize:
            visualize_video(
                scene_name=scene_name,
                inference_visualization_cache=inference_visualization_cache,
                dataset=dataset,
                output_path=args.output_path,
                video_dirpath=args.video_dirpath,
                show_iou=args.visualize_show_iou,
                model_name='KalmanFilter'
            )

    metrics = aggregate_metrics(global_metrics)

    # Save metrics
    logger.info(f'Metrics:\n{json.dumps(metrics, indent=2)}')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    configure_logging(logging.DEBUG)
    main(parse_args())
