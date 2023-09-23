"""
Tracker inference visualization.
"""
import logging
import os

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from collections import Counter

from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets.factory import dataset_factory
from nodetracker.evaluation.end_to_end.config import TrackerGlobalConfig
from nodetracker.evaluation.end_to_end.tracker_utils import TrackerInferenceReader
from nodetracker.library.cv.bbox import PredBBox, BBox
from nodetracker.library.cv.drawing import draw_text
from nodetracker.library.cv import color_palette
from nodetracker.library.cv.video_writer import MP4Writer
from nodetracker.utils import pipeline


logger = logging.getLogger('TrackerVizualization')


NEW_OBJECT_LEN = 5  # Number of frames for which the object is considered `new`


def draw_tracklet(
    frame: np.ndarray,
    tracklet_id: str,
    bbox: PredBBox,
    color = color_palette.RED
) -> np.ndarray:
    """
    Draw tracklet on the frame.

    Args:
        frame: Frame
        tracklet_id: Tracklet id
        bbox: BBox
        color: Bbox color

    Returns:
        Frame with drawn tracklet info
    """
    frame = BBox.draw(bbox, frame, color=color)
    left, top, _, _ = bbox.scaled_yxyx_from_image(frame)
    text = f'[{tracklet_id}] {bbox.label} ({100 * bbox.conf:.0f}%)'
    return draw_text(frame, text, round(left), round(top), color=color)


@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='tracker_visualization', cls=TrackerGlobalConfig)
    cfg: TrackerGlobalConfig
    tracker_output = os.path.join(experiment_path, cfg.tracker.output_path, cfg.eval.split,
                                  cfg.tracker.object_detection.type, cfg.tracker.algorithm.name)
    assert os.path.exists(tracker_output), f'Path "{tracker_output}" does not exist!'
    logger.info(f'Visualizing tracker inference on path "{tracker_output}".')

    dataset = dataset_factory(
        name=cfg.dataset.name,
        path=cfg.dataset.fullpath,
        history_len=1,  # Not relevant
        future_len=1,  # not relevant
        sequence_list=cfg.dataset.split_index[cfg.eval.split],
        additional_params=cfg.dataset.additional_params
    )

    scene_names = dataset.scenes
    for scene_name in tqdm(scene_names, desc='Visualizing tracker', unit='scene'):
        scene_info = dataset.get_scene_info(scene_name)
        scene_length = scene_info.seqlength
        imheight = scene_info.imheight
        imwidth = scene_info.imwidth

        scene_video_path = os.path.join(tracker_output, f'{scene_name}.mp4')
        with TrackerInferenceReader(tracker_output, scene_name, image_height=imheight, image_width=imwidth) as tracker_inf_reader, \
            MP4Writer(scene_video_path, fps=cfg.tracker.visualize.fps) as mp4_writer:
            last_read = tracker_inf_reader.read()
            tracklet_presence_counter = Counter()  # Used to visualize new, appearing tracklets

            for index in tqdm(range(scene_length), desc=f'Visualizing "{scene_name}"', unit='frame'):
                image_path = dataset.get_scene_image_path(scene_name, index)
                frame = cv2.imread(image_path)
                assert frame is not None, \
                    f'Failed to load image for frame {index} on scene "{scene_name}" with path "{image_path}"!'

                if last_read is not None and index == last_read.frame_index:
                    for tracklet_id, bbox in last_read.objects.items():
                        tracklet_presence_counter[tracklet_id] += 1
                        draw_tracklet(
                            frame=frame,
                            tracklet_id=tracklet_id,
                            bbox=bbox,
                            color=color_palette.GREEN if tracklet_presence_counter[tracklet_id] <= NEW_OBJECT_LEN else color_palette.RED
                        )

                    last_read = tracker_inf_reader.read()

                mp4_writer.write(frame)


if __name__ == '__main__':
    main()
