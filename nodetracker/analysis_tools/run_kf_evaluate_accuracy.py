"""
KF motion model evalution
"""
import logging

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets import TorchTrajectoryDataset, dataset_factory
from nodetracker.library.cv.bbox import BBox
from nodetracker.library.kalman_filter.botsort_kf import BotSortKalmanFilter
from nodetracker.utils import pipeline

logger = logging.getLogger('KFMotionModelEvaluation')


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, _ = pipeline.preprocess(cfg, name='dataset_diff_stats')

    dataset = TorchTrajectoryDataset(
        dataset_factory(
            name=cfg.dataset.name,
            path=cfg.dataset.fullpath,
            sequence_list=cfg.dataset.split_index['val'],
            history_len=cfg.dataset.history_len,
            future_len=cfg.dataset.future_len,
            additional_params=cfg.dataset.additional_params
        )
    )

    kf = BotSortKalmanFilter()
    total_iou = 0.0
    total_predictions = 0
    total_successful_matches = 0
    total_match_attempts = 0
    MR_THRESHOLD = 0.35

    for i in tqdm(range(len(dataset)), unit='sample', desc='Evaluating KF motion model'):
        data = dataset[i]
        bboxes_obs, bboxes_unobs = data['orig_bboxes_obs'].numpy(), data['orig_bboxes_unobs'].numpy()

        mean, cov = kf.initiate(bboxes_obs[0])
        for bbox in bboxes_obs[1:]:
            mean, cov = kf.predict(mean, cov)
            mean, cov = kf.update(mean, cov, bbox)

        for j, bbox in enumerate(bboxes_unobs):
            gt_bbox = BBox.from_yxwh(*bbox, clip=True)

            mean, cov = kf.predict(mean, cov)
            mean_projected, _ = kf.project(mean, cov)
            pred_bbox = BBox.from_yxwh(*mean_projected, clip=True)

            iou_score = gt_bbox.iou(pred_bbox)
            total_iou += iou_score
            total_predictions += 1

            if j == bboxes_unobs.shape[0] - 1:
                total_successful_matches += (1 if iou_score >= MR_THRESHOLD else 0)
                total_match_attempts += 1

    accuracy = total_iou / total_predictions
    match_ratio = total_successful_matches / total_match_attempts
    logger.info(f'KF Accuracy: {100 * accuracy:.2f}. Match ratio: {100 * match_ratio:.2f}')


if __name__ == '__main__':
    main()
