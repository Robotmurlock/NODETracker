"""
Visualize script
"""
import logging
import os
from pathlib import Path

import cv2
import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from nodetracker.common import conventions
from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets import dataset_factory
from nodetracker.library.cv import BBox
from nodetracker.utils import pipeline

logger = logging.getLogger('VizScript')


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='visualize')
    assert cfg.visualize is not None, 'Visualize config are not defined!'

    inference_dirpath = conventions.get_inference_path(
        experiment_path=experiment_path,
        model_type=cfg.model.type,
        dataset_name=cfg.dataset.name,
        split=cfg.eval.split,
        experiment_name=cfg.eval.experiment,
        inference_name=cfg.eval.inference_name
    )
    predictions_path = os.path.join(inference_dirpath, 'inference.csv')

    dataset = dataset_factory(
        name=cfg.dataset.name,
        path=cfg.dataset.fullpath,
        sequence_list=cfg.dataset.split_index[cfg.eval.split],
        history_len=cfg.dataset.history_len,
        future_len=cfg.dataset.future_len
    )

    df = pd.read_csv(predictions_path)
    trajectory_groups = df.groupby(['scene_name', 'frame_range'])

    for (scene_name, frame_range), df_traj in tqdm(trajectory_groups, desc='Creating videos', unit='video'):
        df_traj = df_traj.sort_values(by='frame_id')

        mp4_path = conventions.get_inference_video_path(inference_dirpath, scene_name, frame_range)
        Path(mp4_path).parent.mkdir(parents=True, exist_ok=True)
        # noinspection PyUnresolvedReferences
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # noinspection PyUnresolvedReferences
        mp4_writer = cv2.VideoWriter(mp4_path, fourcc, cfg.visualize.fps, cfg.visualize.resolution)

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

            pred_bbox = BBox.from_xyhw(*row[['p_xmin', 'p_ymin', 'p_h', 'p_w']], clip=True)
            frame = pred_bbox.draw(frame, color=(255, 0, 0))

            gt_bbox = BBox.from_xyhw(*row[['gt_xmin', 'gt_ymin', 'gt_h', 'gt_w']], clip=True)
            frame = gt_bbox.draw(frame, color=(0, 0, 255))

        # noinspection PyUnresolvedReferences
        frame = cv2.resize(frame, cfg.visualize.resolution)
        mp4_writer.write(frame)  # write last frame
        logger.debug(f'Saving video ({scene_name}, {frame_range}) at "{mp4_path}"')
        mp4_writer.release()


if __name__ == '__main__':
    main()
