"""
LaSOT sequence analysis - visualization.
"""
import argparse
import logging
import os
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm

from nodetracker.common.project import OUTPUTS_PATH
from nodetracker.datasets.lasot.core import parse_sequence, FrameData
from nodetracker.library.cv import color_palette, drawing
from nodetracker.library.cv.bbox import BBox
from nodetracker.library.cv.video_writer import MP4Writer
from nodetracker.utils.logging import configure_logging

logger = logging.getLogger('LaSOTAnalyzeSequence')


def parse_args() -> argparse.Namespace:
    """
    Sequence analysis - visualization tool configuration.

    Returns:
        Parsed configuration.
    """
    parser = argparse.ArgumentParser(description='Sequence analysis - visualization')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--sequence-name', type=str, required=True, help='Sequence name.')
    parser.add_argument('--output-path', type=str, required=False, default=OUTPUTS_PATH,
                        help='Master path where all outputs are stored.')
    parser.add_argument('--analysis-path', type=str, required=False, default='LaSOT_sequence_analysis',
                        help='Path to the analysis output directory.')
    parser.add_argument('--fps', type=int, required=False, default=30, help='Output video fps.')
    return parser.parse_args()


def visualize_data(data: List[FrameData], text_label: str, sequence_output_video_path: str, fps: int = 30) -> None:
    """
    Visualizes sequence data as video.

    Args:
        data: Parsed sequence data
        text_label: Text label
        sequence_output_video_path: Output sequence video path
        fps: Video fps
    """

    with MP4Writer(sequence_output_video_path, fps=fps) as writer:
        with tqdm(enumerate(data), desc='Drawing video', unit='frame', total=len(data)) as pbar:
            for frame_index, (coords, occlusion, out_of_view, image_path) in pbar:
                timestamp = frame_index / fps
                minute = int(timestamp // 60)
                second = int(timestamp % 60)
                time_label = f'{minute}m {second}s'

                # noinspection PyUnresolvedReferences
                image = cv2.imread(image_path)
                h, w, _ = image.shape

                color = color_palette.CYAN
                if occlusion:
                    logger.info(f'[{time_label}] Occlusion occurred!')
                    color = color_palette.BLUE
                if out_of_view:
                    logger.info(f'[{time_label}] Out of view occurred!')
                    color = color_palette.RED

                coords = [coords[0] / w, coords[1] / h, coords[2] / w, coords[3] / h]
                coords_label = '[' + ', '.join([f'{c:.2f}' for c in coords]) + ']'
                bbox = BBox.from_yxwh(*coords, clip=True)
                image = bbox.draw(image, color=color)
                image = drawing.draw_text(image, coords_label, 5, 15, color=color)
                image = drawing.draw_text(image, f'nlp: {text_label}', 5, 30, color=color)
                image = drawing.draw_text(image, f'occlusion: {occlusion}', 5, 45, color=color)
                image = drawing.draw_text(image, f'out of view: {out_of_view}', 5, 60, color=color)

                writer.write(image)


def main(args: argparse.Namespace) -> None:
    sequence_path = os.path.join(args.dataset_path, args.sequence_name)
    sequence_output_video_path = os.path.join(args.output_path, args.analysis_path, f'{args.sequence_name}.mp4')
    Path(sequence_output_video_path).parent.mkdir(parents=True, exist_ok=True)

    data, text_label = parse_sequence(sequence_path)
    visualize_data(data, text_label, sequence_output_video_path, args.fps)


if __name__ == '__main__':
    configure_logging(logging.INFO)
    main(parse_args())
