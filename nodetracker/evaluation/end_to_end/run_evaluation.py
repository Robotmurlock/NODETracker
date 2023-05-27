import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets import transforms, TrajectoryDataset
from nodetracker.datasets.factory import dataset_factory
from nodetracker.evaluation import sot as sot_metrics
from nodetracker.evaluation.end_to_end import jitter
from nodetracker.evaluation.end_to_end.config import ExtendedE2EGlobalConfig
from nodetracker.evaluation.end_to_end.object_detection import object_detection_inference_factory
from nodetracker.filter import filter_factory, StateModelFilter
from nodetracker.library.cv import drawing, BBox, color_palette
from nodetracker.library.cv.video_writer import MP4Writer
from nodetracker.utils import pipeline
from nodetracker.utils.collections import nesteddict
from tools.utils import create_inference_model

logger = logging.getLogger('E2EFilterEvaluation')

METRICS = [  # TODO: Move to some module
    ('MSE', sot_metrics.mse),
    ('Accuracy', sot_metrics.accuracy),
    ('Success', sot_metrics.success),
    ('NormPrecision', sot_metrics.norm_precision)
]


def aggregate_metrics(sample_metrics: Any) -> Any:
    """
    TODO: Move

    Aggregates dictionary of sample metrics (list of values) to dictionary of metrics.
    All metrics should be "mean-friendly".

    Args:
        sample_metrics: Sample metrics

    Returns:
        Aggregated metrics.
    """
    if isinstance(sample_metrics, list):
        return sum(sample_metrics) / len(sample_metrics)
    elif isinstance(sample_metrics, dict):
        agg = {k: aggregate_metrics(v) for k, v in sample_metrics.items()}
        return dict(sorted(agg.items()))
    else:
        return sample_metrics


def merge_metrics(
        global_metrics: Optional[Dict[str, Any]],
        metrics: Optional[Dict[str, List[float]]],
        scene_name: str,
        category: str
) -> Dict[str, Any]:
    """
    TODO: Move

    Adds aggregated metrics to global metrics dictionary.
    If global dictionary is empty then aggregated dictionary is taken as the global one.
    if added metrics are None then global metrics is returned instead.

    Args:
        global_metrics: Global metrics data
        metrics: Aggregated metrics data (can be empty)
        scene_name: Save scene name metrics
        category: Category

    Returns:
        Merged (global) metrics
    """
    if metrics is None:
        return global_metrics

    if global_metrics is None:
        return {
            'global': metrics,
            'categories': {
                category: metrics
            },
            'scenes': {
                scene_name: metrics
            }
        }

    metric_names = set(global_metrics['global'].keys())
    assert metric_names == set(metrics.keys()), f'Metric keys do not match: {metric_names} != {set(metrics.keys())}'

    # Update category and global
    for name in metric_names:
        global_metrics['global'][name].extend(metrics[name])
        if category not in global_metrics['categories']:
            global_metrics['categories'][category] = {}
        if name not in global_metrics['categories'][category]:
            global_metrics['categories'][category][name] = []
        global_metrics['categories'][category][name].extend(metrics[name])

    # Update scene
    if scene_name not in global_metrics['scenes']:
        global_metrics['scenes'][scene_name] = metrics
    else:
        for name in metric_names:
            global_metrics['scenes'][scene_name][name].extend(metrics[name])

    return global_metrics


def create_filter(cfg: ExtendedE2EGlobalConfig, experiment_path: str) -> StateModelFilter:
    """
    Creates filter based on the given config. Wraps "DIRTY" code.

    Args:
        cfg: Config
        experiment_path: Experiment path

    Returns:
        StateModelFilter
    """
    name = cfg.end_to_end.filter.type
    params = cfg.end_to_end.filter.params

    if name == 'node':
        model = create_inference_model(cfg, experiment_path)
        transform_func = transforms.transform_factory(cfg.transform.name, cfg.transform.params)
        params['model'] = model
        params['transform'] = transform_func

    return filter_factory(name, params)


def visualize_video(
    scene_name: str,
    inference_visualization_cache: Dict[str, Dict[str, Dict[int, List[float]]]],
    dataset: TrajectoryDataset,
    video_dirpath: str,
    model_name: str,
    show_iou: bool = False,
    draw_prior: bool = True,
    draw_posterior: bool = True
) -> None:
    # TODO: Move

    inference_types = list(next(iter(inference_visualization_cache.values())).keys())
    for inference_type in inference_types:
        if inference_type == 'prior' and not draw_prior:
            continue
        if inference_type == 'posterior' and not draw_posterior:
            continue

        full_video_dirpath = os.path.join(video_dirpath, inference_type)
        Path(full_video_dirpath).mkdir(parents=True, exist_ok=True)
        scene_info = dataset.get_scene_info(scene_name)
        n_frames = scene_info.seqlength  # TODO: Works for LaSOT and MOT but may crash for other datasets

        video_path = os.path.join(full_video_dirpath, f'{scene_name}.mp4')

        logger.info(f'Performing visualization for scene "{scene_name}" and inference type "{inference_type}"... '
                    f'Video path is "{video_path}"')
        with MP4Writer(video_path, fps=30) as writer:
            for frame_index in tqdm(range(n_frames), desc='Drawing', unit='frame', total=n_frames):
                image_path = dataset.get_scene_image_path(scene_name, frame_index)
                # noinspection PyUnresolvedReferences
                image = cv2.imread(image_path)

                for object_id, inference_data in inference_visualization_cache.items():
                    gt_data = dataset.get_object_data_label_by_frame_index(object_id, frame_index)
                    if gt_data is not None:
                        gt_bbox = BBox.from_yxwh(*gt_data['bbox'], clip=True)
                        image = gt_bbox.draw(image, color=color_palette.GREEN)

                    pred_data = inference_data[inference_type].get(frame_index)
                    if pred_data is not None:
                        pred_bbox = BBox.from_yxwh(*pred_data, clip=True)
                        image = pred_bbox.draw(image, color=color_palette.RED)

                        if show_iou and gt_data is not None:
                            iou_score = gt_bbox.iou(pred_bbox)
                            left, top, _, _ = pred_bbox.scaled_yxyx_from_image(image)
                            image = drawing.draw_text(image, f'{100 * iou_score:.1f}', left, top - 3, color_palette.RED)

                image = drawing.draw_text(image, 'GroundTruth', 5, 15, color_palette.GREEN)
                image = drawing.draw_text(image, f'{model_name}-{inference_type}', 5, 30, color_palette.RED)

                writer.write(image)


def greedy_pick(
    predictions: torch.Tensor,
    prior_state: Tuple[torch.Tensor, ...],
    n_skipped_detection: int
) -> Tuple[torch.Tensor, bool]:
    max_skip_threshold = 5
    max_dist_threshold = 0.1

    prior_bbox, _ = prior_state
    prior_bbox_center = prior_bbox[:2] + prior_bbox[2:4] / 2

    picked_bbox = None
    min_dist = None
    skip_detection = True
    for p in predictions:
        p_center = p[:2] + p[2:] / 2
        dist = torch.abs(p_center - prior_bbox_center).mean()
        if (n_skipped_detection < max_skip_threshold) and (dist > max_dist_threshold):
            continue

        if min_dist is None or dist < min_dist:
            min_dist = min_dist
            picked_bbox = p
            skip_detection = False

    return picked_bbox, skip_detection


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='e2e_evaluation', cls=ExtendedE2EGlobalConfig)
    cfg: ExtendedE2EGlobalConfig

    dataset = dataset_factory(
        name=cfg.dataset.name,
        path=cfg.dataset.fullpath,
        history_len=1,  # Not relevant
        future_len=1  # not relevant
    )

    smf = create_filter(cfg, experiment_path)
    od_inference = object_detection_inference_factory(
        name=cfg.end_to_end.object_detection.type,
        params=cfg.end_to_end.object_detection.params
    )

    global_metrics = None
    scene_names = dataset.scenes
    if cfg.end_to_end.selection.scene is not None:
        pattern = cfg.end_to_end.selection.scene
        scene_names = [sn for sn in scene_names if re.match(pattern, sn)]

    for scene_name in scene_names:
        object_ids = dataset.get_scene_object_ids(scene_name)
        scene_info = dataset.get_scene_info(scene_name)
        inference_visualization_cache: Dict[str, Dict[str, Dict[int, List[float]]]] = \
            nesteddict()  # Used only for visualization

        for object_id in tqdm(object_ids, unit='track', desc=f'Evaluating tracker on {scene_name}'):
            n_pred_steps = cfg.end_to_end.eval.n_steps
            prior_state, posterior_state, prev_measurement = None, None, None
            first_frame = True

            n_data_points = dataset.get_object_data_length(object_id)
            scene_metrics = defaultdict(list)
            data_points = [dataset.get_object_data_label(object_id, i) for i in range(n_data_points)]
            measurements = [dp['bbox'] for dp in data_points]
            n_skipped_detection = 0

            for index in range(n_data_points - n_pred_steps):
                # Extract point data
                point_data = data_points[index]
                frame_id, image_path = point_data['frame_id'], point_data['image_path']
                measurement = torch.tensor(measurements[index], dtype=torch.float32)
                oov, occ = point_data['oov'], point_data['occ']
                evaluate_step = True

                if first_frame:
                    # First filter iteration
                    posterior_state = smf.initiate(measurement)
                    first_frame = False
                    # Nothing to compare in first iteration (SOT definition)
                    # TODO: This should be configurable in case of MOT
                else:
                    # Perform prediction
                    prior_state = smf.predict(posterior_state)  # Prior
                    prior_multistep_state = smf.multistep_predict(posterior_state, n_pred_steps)

                    # Perform object detection inference
                    predictions = od_inference.predict(point_data)
                    prediction, skip_detection = greedy_pick(predictions, prior_state, n_skipped_detection)

                    # Add evaluation jitter (optional)
                    jitter_skip_detection = jitter.simulate_detector_false_positive(
                        proba=cfg.end_to_end.jitter.detection_skip_proba,
                        first_frame=first_frame
                    )
                    skip_detection = skip_detection or jitter_skip_detection
                    if skip_detection:
                        n_skipped_detection += 1
                    else:
                        n_skipped_detection = 0

                    if not skip_detection:  # Can't perform jitter if detection is missing
                        prediction = jitter.simulate_detector_noise(
                            measurement=prediction,
                            prev_measurement=prev_measurement,
                            sigma=cfg.end_to_end.jitter.detection_noise_sigma
                        )

                    # Check if data point should be used for evaluation
                    if cfg.end_to_end.eval.occlusion_as_skip_detection:
                        skip_detection = skip_detection
                        if oov or occ:
                            skip_detection = skip_detection or oov or occ
                            evaluate_step = False

                    if skip_detection:
                        posterior_state = prior_state
                    else:
                        posterior_state = smf.update(prior_state, prediction)  # Posterior

                    multistep_score = {k: 0.0 for k, _ in METRICS}
                    prior_multistep, _ = smf.project(prior_multistep_state)
                    prior_multistep_numpy = prior_multistep.detach().cpu().numpy()  # Removing batch dimension

                    for p_index in range(n_pred_steps):
                        gt = np.array(measurements[index + p_index], dtype=np.float32)

                        if evaluate_step:
                            for metric_name, metric_func in METRICS:
                                score = metric_func(gt, prior_multistep_numpy[p_index])
                                scene_metrics[f'prior-{metric_name}-{p_index}'].append(score)
                                multistep_score[metric_name] += score

                    if evaluate_step:
                        for metric_name, _ in METRICS:
                            scene_metrics[f'prior-{metric_name}'].append(multistep_score[metric_name] / n_pred_steps)

                    posterior, _ = smf.project(posterior_state)
                    posterior_numpy = posterior.detach().cpu().numpy()

                    if evaluate_step:
                        gt = np.array(measurements[index], dtype=np.float32)
                        for metric_name, metric_func in METRICS:
                            score = metric_func(gt, posterior_numpy)
                            scene_metrics[f'posterior-{metric_name}'].append(score)

                    # Save data
                    prev_measurement = measurement.detach().cpu().numpy()

                    if cfg.end_to_end.visualization.enable:
                        prior, _ = smf.project(prior_state)
                        prior_numpy = prior.detach().cpu().numpy()
                        inference_visualization_cache[object_id]['prior'][frame_id] = prior_numpy.tolist()
                        inference_visualization_cache[object_id]['posterior'][frame_id] = posterior_numpy.tolist()

            global_metrics = merge_metrics(
                global_metrics=global_metrics,
                metrics=scene_metrics,
                scene_name=scene_name,
                category=scene_info.category
            )

        if cfg.end_to_end.visualization:
            video_dirpath = os.path.join(experiment_path, 'visualization',
                                         f'end_to_end_{cfg.end_to_end.filter.type}')
            visualize_video(
                scene_name=scene_name,
                inference_visualization_cache=inference_visualization_cache,
                dataset=dataset,
                video_dirpath=video_dirpath,
                model_name=smf.__class__.__name__,
                show_iou=cfg.end_to_end.visualization.show_iou,
                draw_prior=cfg.end_to_end.visualization.prior,
                draw_posterior=cfg.end_to_end.visualization.posterior
            )

    metrics = aggregate_metrics(global_metrics)

    # Save metrics
    logger.info(f'Metrics: \n{json.dumps(metrics, indent=2)}')
    metrics_path = os.path.join(experiment_path, 'evaluation', f'end_to_end_{cfg.end_to_end.filter.type}.json')
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
