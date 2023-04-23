"""
LaSOT sequence analysis - visualization.
"""
import argparse
import os
from pathlib import Path
from typing import Tuple, List

import cv2
from tqdm import tqdm

from nodetracker.common.project import OUTPUTS_PATH
from nodetracker.library.cv import color_palette, drawing
from nodetracker.library.cv.bbox import BBox
from nodetracker.library.cv.video_writer import MP4Writer
from nodetracker.utils import file_system
from nodetracker.utils.logging import configure_logging
import logging


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
    return parser.parse_args()


FrameData = Tuple[List[int], bool, bool, str]  # coords, occlusion, out of view, image path


def parse_sequence(sequence_path: str) -> Tuple[List[FrameData], str]:
    """
    Parses sequence directory. Sequence data contains info for each frame:
    - bbox coordinates
    - is object occluded
    - is object out of view

    Sequence also contains text label.

    Args:
        sequence_path: Sequence path

    Returns:
        Parsed sequence with data:
        - Sequence data (bbox, occlusions, out of view)
        - text label
        - path to sequence images
    """
    occlusion_filepath = os.path.join(sequence_path, 'full_occlusion.txt')
    out_of_view_filepath = os.path.join(sequence_path, 'out_of_view.txt')
    nlp_filepath = os.path.join(sequence_path, 'nlp.txt')
    gt_filepath = os.path.join(sequence_path, 'groundtruth.txt')

    with open(occlusion_filepath, 'r', encoding='utf-8') as f:
        occlusion_raw = f.read()
        occlusions = [bool(int(o)) for o in occlusion_raw.strip().split(',')]

    with open(out_of_view_filepath, 'r', encoding='utf-8') as f:
        oov_raw = f.read()
        oov = [bool(int(o)) for o in oov_raw.strip().split(',')]

    with open(nlp_filepath, 'r', encoding='utf-8') as f:
        text_label = f.read().strip()

    with open(gt_filepath, 'r', encoding='utf-8') as f:
        coord_lines = f.readlines()
        coords = [[int(v) for v in c.strip().split(',')] for c in coord_lines]

    image_dirpath = os.path.join(sequence_path, 'img')
    image_paths = [os.path.join(image_dirpath, image_name) for image_name in sorted(file_system.listdir(image_dirpath))]

    assert len(occlusions) == len(oov) == len(coords), 'Failed to parse sequence!'
    data = list(zip(coords, occlusions, oov, image_paths))
    return data, text_label


def visualize_data(data: List[FrameData], text_label: str, sequence_output_video_path: str) -> None:
    """
    Visualizes sequence data as video.

    Args:
        data: Parsed sequence data
        text_label: Text label
        sequence_output_video_path: Output sequence video path
    """
    FPS = 30

    with MP4Writer(sequence_output_video_path, fps=FPS) as writer:
        for frame_index, (coords, occlusion, out_of_view, image_path) in tqdm(enumerate(data),
                                                                              desc='Drawing video', unit='frame', total=len(data)):
            timestamp = frame_index / FPS
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
    visualize_data(data, text_label, sequence_output_video_path)


if __name__ == '__main__':
    configure_logging(logging.INFO)
    main(parse_args())
