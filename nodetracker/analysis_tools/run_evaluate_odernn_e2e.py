"""
Evaluate ODERNN E2E
"""
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from nodetracker.analysis_tools.run_evaluate_kf_e2e import aggregate_metrics, calc_iou, \
    simulate_detector_noise, simulate_detector_false_positive
from nodetracker.common import conventions
from nodetracker.common.project import CONFIGS_PATH
from nodetracker.common.project import OUTPUTS_PATH
from nodetracker.datasets import transforms, dataset_factory
from nodetracker.node import load_or_create_model, LightningGaussianModel
from nodetracker.utils import pipeline

logger = logging.getLogger('ODERNN_E2E_EVAL')


# Improvisation
N_STEPS: int = 5
N_HIST: int = 10
DETECTION_NOISE_SIGMA: float = 0.00
DETECTION_NOISE: int = 4 * DETECTION_NOISE_SIGMA * torch.ones(4, dtype=torch.float32)
N_MAX_OBJS: Optional[int] = None
DET_SKIP_PROBA: float = 0.0


class ODETorchTensorBuffer:
    def __init__(self, size: int, min_size: int, dtype: torch.dtype):
        assert size >= 1, f'Invalid size {size}. Minimum size is 1.'

        self._size = size
        self._min_size = min_size
        self._dtype = dtype

        self._buffer: List[torch.Tensor] = []

    @property
    def has_input(self) -> bool:
        return len(self._buffer) >= self._min_size

    def push(self, x: torch.Tensor) -> None:
        self._buffer.append(x)
        if len(self._buffer) > self._size:
            self._buffer.pop(0)

    def get_input(self, n_future_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_hist_steps = len(self._buffer)
        x_obs = torch.stack(self._buffer).view(n_hist_steps, 1, -1)
        ts_obs = torch.tensor(list(range(1, n_hist_steps + 1)), dtype=torch.float32).view(-1, 1, 1)
        ts_unobs = torch.tensor(list(range(n_hist_steps + 1, n_hist_steps + n_future_steps + 1)), dtype=torch.float32).view(-1, 1, 1)

        return x_obs, ts_obs, ts_unobs


class ODERNNFilter:
    def __init__(self, model: LightningGaussianModel, transform: transforms.InvertibleTransformWithStd, accelerator: str):
        self._model = model
        self._transform = transform

        self._accelerator = accelerator
        self._model.to(self._accelerator)
        self._model.eval()

    def predict(self, x_obs: torch.Tensor, ts_obs: torch.Tensor, ts_unobs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_x_obs, _, t_ts_obs, *_ = self._transform.apply(data=(x_obs, None, ts_obs), shallow=False)
        t_x_obs, t_ts_obs, ts_unobs = t_x_obs.to(self._accelerator), t_ts_obs.to(self._accelerator), ts_unobs.to(self._accelerator)
        t_x_unobs_mean_hat, t_x_unobs_std_hat, *_ = self._model.inference(t_x_obs, t_ts_obs, ts_unobs)
        t_x_unobs_mean_hat, t_x_unobs_std_hat = t_x_unobs_mean_hat.detach().cpu(), t_x_unobs_std_hat.detach().cpu()
        _, x_unobs_mean_hat, *_ = self._transform.inverse(data=[x_obs, t_x_unobs_mean_hat], shallow=True)
        x_unobs_std_hat = self._transform.inverse_std(t_x_unobs_std_hat)

        return x_unobs_mean_hat, x_unobs_std_hat

    # noinspection PyMethodMayBeStatic
    def update(self, x_unobs_mean_hat: torch.Tensor, x_unobs_std_hat: torch.Tensor, det_mean: torch.Tensor, det_std: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        def std_inv(m: torch.Tensor) -> torch.Tensor:
            return (1 / m).nan_to_num(nan=0, posinf=0, neginf=0)

        x_unobs_cov_hat_inv = std_inv(x_unobs_std_hat)
        det_cov_inv = std_inv(det_std)
        gain = std_inv(x_unobs_cov_hat_inv + det_cov_inv) * det_cov_inv

        innovation = (det_mean - x_unobs_mean_hat)
        final_mean = x_unobs_mean_hat + gain * innovation
        final_std = (1 - gain) * x_unobs_std_hat

        return final_mean, final_std


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='visualize_trajectories')

    dataset = dataset_factory(
        name=cfg.dataset.name,
        path=cfg.dataset.fullpath,
        sequence_list=cfg.dataset.split_index[cfg.eval.split],
        history_len=1,  # Not relevant
        future_len=1  # not relevant
    )

    transform_func = transforms.transform_factory(cfg.transform.name, cfg.transform.params)

    sample_metrics = defaultdict(list)

    Path(OUTPUTS_PATH).mkdir(parents=True, exist_ok=True)
    metrics_path = os.path.join(OUTPUTS_PATH,
                                f'odernn_metrics_noise{DETECTION_NOISE_SIGMA}_skip{DET_SKIP_PROBA}.json')
    logger.info('Evaluating ODERNN filter')

    checkpoint_path = conventions.get_checkpoint_path(experiment_path, cfg.eval.checkpoint) \
        if cfg.eval.checkpoint else None
    model = load_or_create_model(
        model_type=cfg.model.type,
        params=cfg.model.params,
        checkpoint_path=checkpoint_path
    )
    assert isinstance(model, LightningGaussianModel), \
        'Uncertainty measurement is only available for ODERNN, RNNODE and MLPODE (instances of LightningGaussianModel)!'
    assert model.is_modeling_gaussian, 'Uncertainty measurement is only available for ODERNN trained with GaussianNLLLoss!'
    accelerator = cfg.resources.accelerator
    model.to(accelerator)

    ode_filter = ODERNNFilter(model=model, transform=transform_func, accelerator=cfg.resources.accelerator)
    n_pred_steps = N_STEPS

    n_finished_objs = 0

    scene_names = dataset.scenes
    for scene_name in scene_names:
        object_ids = dataset.get_scene_object_ids(scene_name)
        for object_id in tqdm(object_ids, unit='track', desc=f'Evaluating tracker on {scene_name}'):
            buffer = ODETorchTensorBuffer(N_HIST, min_size=10, dtype=torch.float32)
            prev_measurement, x_mean_hat, x_std_hat, mean, covariance = None, None, None, None, None

            n_data_points = dataset.get_object_data_length(object_id)
            for index in range(n_data_points-n_pred_steps):
                measurement = dataset.get_object_data_label(object_id, index)['bbox']
                measurement_no_noise = torch.tensor(measurement, dtype=torch.float32)
                measurement = simulate_detector_noise(measurement, prev_measurement, DETECTION_NOISE_SIGMA)
                skip_detection = simulate_detector_false_positive(DET_SKIP_PROBA, first_frame=not buffer.has_input)

                measurement = torch.tensor(measurement, dtype=torch.float32)

                if buffer.has_input:
                    x_obs, ts_obs, ts_unobs = buffer.get_input(n_pred_steps)
                    x_mean_hat, x_std_hat = ode_filter.predict(x_obs, ts_obs, ts_unobs)
                    detection_noise = torch.tensor(prev_measurement) * DETECTION_NOISE
                    x_mean_hat_first, x_std_hat_first = x_mean_hat[0, 0, :], x_std_hat[0, 0, :]

                    if skip_detection:
                        mean, covariance = x_mean_hat_first, x_std_hat_first
                    else:
                        mean, covariance = ode_filter.update(x_mean_hat_first, x_std_hat_first,
                                                             measurement, detection_noise)

                    total_mse = 0.0
                    total_iou = 0.0
                    for p_index in range(n_pred_steps):
                        pred = x_mean_hat[p_index, 0]
                        gt = torch.tensor(dataset.get_object_data_label(object_id, index + p_index)['bbox'],
                                          dtype=torch.float32)

                        # Calculate MSE
                        mse = ((gt - pred) ** 2).mean().item()
                        sample_metrics[f'prior-MSE-{p_index}'].append(mse)
                        total_mse += mse

                        # Calculate IOU (Accuracy)
                        iou_score = calc_iou(pred, gt)
                        sample_metrics[f'prior-Accuracy-{p_index}'].append(iou_score)
                        total_iou += iou_score

                    sample_metrics['prior-MSE'].append(total_mse / n_pred_steps)
                    sample_metrics['prior-Accuracy'].append(total_iou / n_pred_steps)

                    posterior_mse = ((mean - measurement_no_noise) ** 2).mean().item()
                    sample_metrics[f'posterior-MSE'].append(posterior_mse)
                    posterior_iou_score = calc_iou(mean, measurement_no_noise)
                    sample_metrics[f'posterior-Accuracy'].append(posterior_iou_score)

                if skip_detection:
                    buffer.push(mean.clone())
                else:
                    buffer.push(measurement.clone())

                # Save data
                prev_measurement = measurement.detach().cpu().numpy().tolist()

            n_finished_objs += 1
            if N_MAX_OBJS is not None and n_finished_objs >= N_MAX_OBJS:
                break

    metrics = aggregate_metrics(sample_metrics)

    # Save metrics
    logger.info(f'Metrics: \n{json.dumps(metrics, indent=2)}')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
