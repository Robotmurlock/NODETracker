"""
Evaluate Kalman Filter
"""
import argparse
import json
import logging
import os
from pathlib import Path

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
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    dataset_path = os.path.join(args.input_path, args.dataset_name, args.split)
    logger.info(f'Loading dataset from path "{dataset_path}"')
    dataset = MOTDataset(
        path=dataset_path,
        history_len=1,  # Not relevant
        future_len=1,  # not relevant
        label_type=LabelType.GROUND_TRUTH
    )

    mse_results = []
    iou_results = []

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    inference_path = os.path.join(args.output_path, 'kf_inference.csv')
    metrics_path = os.path.join(args.output_path, 'kf_metrics.json')

    logger.info('Writing inference data.')
    with open(inference_path, 'w', encoding='utf-8') as writer:
        header = 'scene,object_id,gt_ymin,gt_xmin,gt_w,gt_h,pred_ymin,pred_xmin,pred_w,pred_h'
        writer.write(header)

        scene_names = dataset.scenes
        for scene_name in scene_names:
            object_ids = dataset.get_scene_object_ids(scene_name)
            for object_id in tqdm(object_ids, unit='track', desc=f'Evaluating tracker on {scene_name}'):
                _, object_scene_id = dataset.parse_object_id(object_id)
                kf = BotSortKalmanFilter()  # Create KF for new track
                first_iteration = True
                mean, covariance, mean_hat, covariance_hat = None, None, None, None
                gt, gt_ymin, gt_xmin, gt_w, gt_h = None, None, None, None, None

                n_data_points = dataset.get_object_data_length(object_id)
                for index in range(n_data_points):
                    measurement = dataset.get_object_data_label(object_id, index)['bbox']

                    if mean is None:
                        # First KF iteration
                        mean, covariance = kf.initiate(measurement)
                        pred, _ = kf.project(mean, covariance)
                    else:
                        mean_hat, covariance_hat = kf.predict(mean, covariance)
                        mean, covariance = kf.update(mean_hat, covariance_hat, measurement)
                        pred, _ = kf.project(mean_hat, covariance_hat)

                    if not first_iteration:
                        # Nothing to compare in first iteration
                        pred_ymin, pred_xmin, pred_w, pred_h = pred
                        gt_bbox = BBox.from_xyhw(gt_xmin, gt_ymin, gt_h, gt_w, clip=True)
                        pred_bbox = BBox.from_xyhw(pred_xmin, pred_ymin, pred_h, pred_w, clip=True)

                        mse = ((gt - pred) ** 2).mean()
                        iou_score = gt_bbox.iou(pred_bbox)
                        mse_results.append(mse)
                        iou_results.append(iou_score)

                        row_data = \
                            f'\n{scene_name},{object_scene_id},{index},{gt_ymin},{gt_xmin},{gt_w},{gt_h},{pred_ymin},{pred_xmin},{pred_w},{pred_h}'
                        writer.write(row_data)
                    else:
                        first_iteration = False

                    # Ground truth values need to be 1 step behind in order to compare with predictions
                    gt_ymin, gt_xmin, gt_w, gt_h = measurement
                    gt = measurement.copy()

    metrics = {
        'mse': sum(mse_results) / len(mse_results),
        'avg_iou': sum(iou_results) / len(iou_results)
    }
    logger.info(f'Metrics: \n{json.dumps(metrics, indent=2)}')

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    configure_logging(logging.DEBUG)
    main(parse_args())
