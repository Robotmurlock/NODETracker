"""
Training script
"""
import logging
import os
import cv2

import hydra
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from nodetracker.common import conventions
from nodetracker.common.project import CONFIGS_PATH
from nodetracker.utils import pipeline
from nodetracker.datasets import MOTDataset

logger = logging.getLogger('VizScript')


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='visualize')
    assert cfg.visualize is not None, 'Visualize config are not defined!'

    inference_dirpath = conventions.get_inference_path(experiment_path, cfg.model.type, cfg.dataset.name, cfg.eval.split, cfg.eval.experiment,
                                                       cfg.eval.inference_name)
    predictions_path = os.path.join(inference_dirpath, 'inference.csv')

    dataset_path = os.path.join(cfg.path.assets, cfg.dataset.get_split_path(cfg.eval.split))
    logger.info(f'Dataset {cfg.eval.split} path: "{dataset_path}".')

    dataset = MOTDataset(
        path=dataset_path,
        history_len=cfg.dataset.history_len,
        future_len=cfg.dataset.future_len
    )

    df = pd.read_csv(predictions_path)
    trajectory_groups = df.groupby(['scene_name', 'frame_range'])
    for (scene_name, frame_range), df_traj in tqdm(trajectory_groups, desc='Creating videos', unit='video'):
        df_traj = df_traj.sort_values(by='frame_id')
        scene_info = dataset.get_scene_info(scene_name)
        h, w = scene_info.imheight, scene_info.imwidth

        mp4_path = conventions.get_inference_video_path(inference_dirpath, scene_name, frame_range)
        Path(mp4_path).parent.mkdir(parents=True, exist_ok=True)
        # noinspection PyUnresolvedReferences
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # noinspection PyUnresolvedReferences
        mp4_writer = cv2.VideoWriter(mp4_path, fourcc, cfg.visualize.fps, (600, 400))

        frame_id, frame_path, frame = None, None, None
        for _, row in df_traj.iterrows():
            row_frame_id = row['frame_id']
            if frame_id != row_frame_id:
                if frame is not None:
                    # noinspection PyUnresolvedReferences
                    frame = cv2.resize(frame, cfg.visualize.resolution)
                    mp4_writer.write(frame)

                frame_id = int(row_frame_id)
                frame_path = dataset.get_scene_image_path(scene_name, frame_id)
                # noinspection PyUnresolvedReferences
                frame = cv2.imread(frame_path)

            p_ymin, p_xmin, p_w, p_h = row[['p_ymin', 'p_xmin', 'p_w', 'p_h']]
            p_yc, p_xc = int((p_ymin + p_w / 2) * w), int((p_xmin + p_h / 2) * h)
            # noinspection PyUnresolvedReferences
            frame = cv2.circle(frame, (p_yc, p_xc), radius=3, color=(255, 0, 0), thickness=3)

            gt_ymin, gt_xmin, gt_w, gt_h = row[['gt_ymin', 'gt_xmin', 'gt_w', 'gt_h']]
            gt_yc, gt_xc = int((gt_ymin + gt_w / 2) * w), int((gt_xmin + gt_h / 2) * h)
            # noinspection PyUnresolvedReferences
            frame = cv2.circle(frame, (gt_yc, gt_xc), radius=3, color=(0, 0, 255), thickness=3)

        # noinspection PyUnresolvedReferences
        frame = cv2.resize(frame, cfg.visualize.resolution)
        mp4_writer.write(frame)  # write last frame
        logger.debug(f'Saving video ({scene_name}, {frame_range}) at "{mp4_path}"')
        mp4_writer.release()
        break


if __name__ == '__main__':
    main()
