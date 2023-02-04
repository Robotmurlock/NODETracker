"""
Evaluate Kalman Filter
"""
import argparse
import json
import logging
import os
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from nodetracker.common.project import ASSETS_PATH, OUTPUTS_PATH
from nodetracker.datasets.mot.core import MOTDataset, LabelType
from nodetracker.library.cv.bbox import BBox
from nodetracker.library.kalman_filter.botsort_kf import BotSortKalmanFilter
from nodetracker.utils.logging import configure_logging

logger = logging.getLogger('MotTools')


def parse_args() -> argparse.Namespace:
    """
    Returns:
        Parse tool arguments
    """
    parser = argparse.ArgumentParser(description='Kalman Filter evaluation')
    parser.add_argument('--input-path', type=str, required=False, default=ASSETS_PATH, help='Datasets path')
    parser.add_argument('--output-path', type=str, required=False, default=OUTPUTS_PATH, help='Output path')
    parser.add_argument('--split', type=str, required=False, default='val', help='Dataset split name')
    parser.add_argument('--dataset-name', type=str, required=True, help='Dataset name')
    parser.add_argument('--steps', type=int, required=False, default=1, help='Number of forecast steps')
    return parser.parse_args()


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
            _, object_scene_id = dataset.parse_object_id(object_id)
            kf = BotSortKalmanFilter()  # Create KF for new track
            mean, covariance, mean_hat, covariance_hat = None, None, None, None
            gt, gt_ymin, gt_xmin, gt_w, gt_h = None, None, None, None, None

            n_data_points = dataset.get_object_data_length(object_id)
            for index in range(n_data_points):
                measurement = dataset.get_object_data_label(object_id, index)['bbox']

                if mean is None:
                    # First KF iteration
                    mean, covariance = kf.initiate(measurement)
                    pred, _ = kf.project(mean, covariance)
                    # Nothing to compare in first iteration
                else:
                    # Perform prediction
                    mean_hat, covariance_hat = kf.predict(mean, covariance)
                    mean, covariance = kf.update(mean_hat, covariance_hat, measurement)

                    total_mse = 0.0
                    total_iou = 0.0
                    for p_index in range(1, n_pred_steps+1):
                        pred, _ = kf.project(mean_hat, covariance_hat)  # KF Prediction is before update

                        # Calculate MSE
                        mse = ((gt - pred) ** 2).mean()
                        sample_metrics[f'MSE-{p_index}'].append(mse)
                        total_mse += mse

                        # Calculate IOU (Accuracy)
                        pred_ymin, pred_xmin, pred_w, pred_h = pred
                        gt_bbox = BBox.from_xyhw(gt_xmin, gt_ymin, gt_h, gt_w, clip=True)
                        pred_bbox = BBox.from_xyhw(pred_xmin, pred_ymin, pred_h, pred_w, clip=True)

                        iou_score = gt_bbox.iou(pred_bbox)
                        sample_metrics[f'Accuracy-{p_index}'].append(iou_score)
                        total_iou += iou_score

                        mean_hat, covariance_hat = kf.predict(mean_hat, covariance_hat)

                    sample_metrics['MSE'].append(total_mse / n_pred_steps)
                    sample_metrics['Accuracy'].append(total_iou / n_pred_steps)

                # Ground truth values need to be 1 step behind in order to compare with predictions
                gt_ymin, gt_xmin, gt_w, gt_h = measurement
                gt = measurement.copy()

    metrics = {k: sum(v) / len(v) for k, v in sample_metrics.items()}  # Metrics aggregation (averaging)
    logger.info(f'Metrics: \n{json.dumps(metrics, indent=2)}')

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    configure_logging(logging.DEBUG)
    main(parse_args())
