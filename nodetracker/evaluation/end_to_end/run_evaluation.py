import copy
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
from nodetracker.library.cv import drawing, BBox, color_palette, PredBBox
from nodetracker.library.cv.video_writer import MP4Writer
from nodetracker.utils import pipeline
from nodetracker.utils.collections import nesteddict
from nodetracker.utils.lookup import LookupTable
from tools.utils import create_inference_model
from nodetracker.evaluation.end_to_end.inference_writer import InferenceWriter

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
            'global': copy.deepcopy(metrics),
            'categories': {
                category: copy.deepcopy(metrics)
            },
            'scenes': {
                scene_name: copy.deepcopy(metrics)
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
    inference_visualization_cache: Dict[str, Dict[str, Dict[int, list]]],
    dataset: TrajectoryDataset,
    video_dirpath: str,
    model_name: str,
    show_iou: bool = False,
    draw_prior: bool = True,
    draw_posterior: bool = True
) -> None:
    # TODO: Move

    for inference_type in ['prior', 'posterior']:
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
            ma_iou_score = None  # Moving Average IOU Score
            for frame_index in tqdm(range(n_frames), desc='Drawing', unit='frame', total=n_frames):
                image_path = dataset.get_scene_image_path(scene_name, frame_index)
                # noinspection PyUnresolvedReferences
                image = cv2.imread(image_path)

                # Pack description that will be shown in top left (text, color, order)
                description_text: List[Tuple[str, color_palette.ColorType, int]] = []

                for object_id, inference_data in inference_visualization_cache.items():
                    # Visualize OD inference
                    if frame_index in inference_data['inference_bboxes']:
                        bboxes = inference_data['inference_bboxes'][frame_index]
                        categories = inference_data['inference_cls'][frame_index]
                        confs = inference_data['inference_conf'][frame_index]
                        for bbox, category, conf in zip(bboxes, categories, confs):
                            pred_bbox = PredBBox.create(
                                bbox=BBox.from_yxwh(*bbox, clip=True),
                                label=category,
                                conf=conf
                            )
                            image = pred_bbox.draw(image, color=color_palette.BLUE)

                    # Visualize ground truth
                    gt_data = dataset.get_object_data_label_by_frame_index(object_id, frame_index)
                    if gt_data is not None:
                        gt_bbox = BBox.from_yxwh(*gt_data['bbox'], clip=True)
                        image = gt_bbox.draw(image, color=color_palette.GREEN)

                    # Visualize filter prediction/estimation
                    pred_data = inference_data[inference_type].get(frame_index)
                    if pred_data is not None:
                        pred_bbox = BBox.from_yxwh(*pred_data, clip=True)
                        image = pred_bbox.draw(image, color=color_palette.RED)

                        if show_iou and gt_data is not None:
                            iou_score = gt_bbox.iou(pred_bbox)
                            left, top, _, _ = pred_bbox.scaled_yxyx_from_image(image)
                            ma_iou_score = iou_score if ma_iou_score is None \
                                else 0.9 * ma_iou_score + 0.1 * iou_score

                            iou_text = f'CurrentAccuracy = {100 * iou_score:.1f}%'
                            ma_iou_text = f'MovingAverageAccuracy = {100 * ma_iou_score:.1f}%'
                            description_text.append((iou_text, color_palette.RED, 201))
                            description_text.append((ma_iou_text, color_palette.RED, 202))

                description_text.append(('GroundTruth', color_palette.GREEN, 101))
                description_text.append(('ObjectDetection', color_palette.BLUE, 102))
                description_text.append((f'{model_name}-{inference_type}', color_palette.RED, 103))

                # Draw description text
                description_text = sorted(description_text, key=lambda x: x[2])
                for i, (text, color, _) in enumerate(description_text):
                    image = drawing.draw_text(image, text, 5, 15 * (i + 1), color)

                writer.write(image)


def greedy_pick(
    predictions: torch.Tensor,
    prior_state: Tuple[torch.Tensor, ...],
    n_skipped_detection: int
) -> Tuple[torch.Tensor, bool]:
    max_skip_threshold = 5
    min_iou_match = 0.3

    prior_bbox, _ = prior_state
    prior_bbox_object = BBox.from_xyhw(*prior_bbox[:4], clip=True)

    picked_bbox = None
    max_iou = None
    skip_detection = True
    for p in predictions:
        p_bbox = BBox.from_xyhw(*p[:4], clip=True)
        iou_score = prior_bbox_object.iou(p_bbox)

        if (n_skipped_detection < max_skip_threshold) and (iou_score < min_iou_match):
            continue

        if max_iou is None or iou_score > max_iou:
            max_iou = iou_score
            picked_bbox = p
            skip_detection = False

    return picked_bbox, skip_detection


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='e2e_evaluation', cls=ExtendedE2EGlobalConfig)
    cfg: ExtendedE2EGlobalConfig

    with open(cfg.end_to_end.lookup_path, 'r', encoding='utf-8') as f:
        lookup = LookupTable.deserialize(json.load(f))

    dataset = dataset_factory(
        name=cfg.dataset.name,
        path=cfg.dataset.fullpath,
        history_len=1,  # Not relevant
        future_len=1,  # not relevant
        sequence_list=cfg.dataset.split_index[cfg.eval.split] if cfg.end_to_end.selection.eval_split_only else None
    )

    od_inference = object_detection_inference_factory(
        name=cfg.end_to_end.object_detection.type,
        params=cfg.end_to_end.object_detection.params,
        lookup=lookup
    )

    global_metrics = None
    scene_names = dataset.scenes
    if cfg.end_to_end.selection.scene is not None:
        pattern = cfg.end_to_end.selection.scene
        scene_names = [sn for sn in scene_names if re.match(pattern, sn)]

    model_name: Optional[str] = None
    for scene_name in scene_names:
        object_ids = dataset.get_scene_object_ids(scene_name)
        inf_viz_cache: Dict[str, Dict[str, Dict[int, list]]] = \
            nesteddict()  # Used only for visualization

        for object_id in tqdm(object_ids, unit='track', desc=f'Evaluating tracker on {scene_name}'):
            smf = create_filter(cfg, experiment_path)
            model_name = smf.__class__.__name__

            n_pred_steps = cfg.end_to_end.eval.n_steps
            prior_state, posterior_state, prev_measurement = None, None, None
            first_frame = True

            n_data_points = dataset.get_object_data_length(object_id)
            scene_metrics = defaultdict(list)
            data_points = [dataset.get_object_data_label(object_id, i) for i in range(n_data_points)]
            measurements = [dp['bbox'] for dp in data_points]
            n_skipped_detection = 0

            inference_writer = None
            if cfg.end_to_end.save_inference:
                object_inference_filepath = os.path.join(experiment_path, 'evaluation', 'object_inference', f'{object_id}.csv')
                inference_writer = InferenceWriter(object_inference_filepath)
                inference_writer.open()

            for index in range(n_data_points - n_pred_steps):
                # Extract point data
                point_data = data_points[index]
                frame_id = point_data['frame_id']
                measurement = torch.tensor(measurements[index], dtype=torch.float32)
                measurement_numpy = measurement.detach().cpu().numpy()
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
                    inf_bboxes, inf_classes, inf_conf = od_inference.predict(point_data)
                    bboxes, skip_detection = greedy_pick(inf_bboxes, prior_state, n_skipped_detection)

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
                        bboxes = jitter.simulate_detector_noise(
                            measurement=bboxes,
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
                        posterior_state = smf.missing(prior_state)
                    else:
                        posterior_state = smf.update(prior_state, bboxes)

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

                    prior, _ = smf.project(prior_state)
                    prior_numpy = prior.detach().cpu().numpy()
                    inf_bboxes_numpy = inf_bboxes.detach().cpu().numpy()
                    inf_conf_numpy = inf_conf.detach().cpu().numpy()
                    bboxes_numpy = bboxes.detach().cpu().numpy() if bboxes is not None else np.array([None] * 4, dtype=np.float32)

                    if cfg.end_to_end.visualization.enable:
                        inf_viz_cache[object_id]['inference_bboxes'][frame_id] = inf_bboxes_numpy.tolist()
                        inf_viz_cache[object_id]['inference_cls'][frame_id] = inf_classes
                        inf_viz_cache[object_id]['inference_conf'][frame_id] = inf_conf_numpy.tolist()
                        inf_viz_cache[object_id]['prior'][frame_id] = prior_numpy.tolist()
                        inf_viz_cache[object_id]['posterior'][frame_id] = posterior_numpy.tolist()

                    if inference_writer is not None:
                        inference_writer.write(
                            frame_index=frame_id,
                            prior=prior_numpy,
                            posterior=posterior_numpy,
                            ground_truth=measurement_numpy,
                            od_prediction=bboxes_numpy,
                            occ=occ,
                            oov=oov
                        )

            global_metrics = merge_metrics(
                global_metrics=global_metrics,
                metrics=scene_metrics,
                scene_name=scene_name,
                category=dataset.get_object_category(object_id)
            )

            if inference_writer is not None:
                inference_writer.close()

        if cfg.end_to_end.visualization.enable:
            video_dirpath = os.path.join(experiment_path, 'visualization',
                                         f'end_to_end_{cfg.end_to_end.filter.type}')
            visualize_video(
                scene_name=scene_name,
                inference_visualization_cache=inf_viz_cache,
                dataset=dataset,
                video_dirpath=video_dirpath,
                model_name=model_name,
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
