import numpy as np
import os
from pathlib import Path


def get_inference_path(experiment_path: str, filter_type: str) -> str:
    return os.path.join(experiment_path, 'evaluation', filter_type, 'object_inference')


class InferenceWriter:
    def __init__(self, path: str):
        self._path = path
        self._header = ','.join([
            'frame',
            'prior_y', 'prior_x', 'prior_w', 'prior_h',
            'posterior_y', 'posterior_x', 'posterior_w', 'posterior_h',
            'gt_y', 'gt_x', 'gt_w', 'gt_h',
            'od_pred_y', 'od_pred_x', 'od_pred_w', 'od_pred_h',
            'occlusion', 'out_of_view',
            'step_iou', 'prior_iou', 'posterior_iou'
        ])
        self._header_sep_cnt = self._header.count(',')

        # State
        self._writer = None

    def open(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._writer = open(self._path, 'w', encoding='utf-8')
        self._writer.write(self._header)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    def write(
        self,
        frame_index: int,
        prior: np.ndarray,
        posterior: np.ndarray,
        ground_truth: np.ndarray,
        od_prediction: np.ndarray,
        occ: bool,
        oov: bool,
        step_iou: float,
        prior_iou: float,
        posterior_iou: float
    ):
        for vector in [prior, posterior, ground_truth, od_prediction]:
            assert vector.shape == (4,), f'Expected vector shape (4,) but found {vector.shape}!'

        data = [str(frame_index)] + \
               [str(v) for v in prior] + \
               [str(v) for v in posterior] + \
               [str(v) for v in ground_truth] + \
               [str(v) for v in od_prediction] + \
               [str(int(occ)), str(int(oov))] + \
               [f'{100 * step_iou:.2f}', f'{100 * prior_iou:.2f}', f'{100 * posterior_iou:.2f}']

        line = ','.join(data)
        line_sep_cnt = line.count(',')
        assert self._header_sep_cnt == line_sep_cnt, f'Expected {self._header_sep_cnt + 1} columns but found {line_sep_cnt + 1} columns!'
        self._writer.write(os.linesep)
        self._writer.write(line)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
