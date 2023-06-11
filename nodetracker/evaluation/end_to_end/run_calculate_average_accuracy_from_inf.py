"""
Calculate Average Accuracy from inference (csv) dump.
"""
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from nodetracker.common.project import CONFIGS_PATH
from nodetracker.datasets import dataset_factory
from nodetracker.evaluation.end_to_end.config import ExtendedE2EGlobalConfig
from nodetracker.evaluation.end_to_end.inference_writer import get_inference_path
from nodetracker.utils import pipeline, file_system


def average_accuracy(
    df: pd.DataFrame,
    n_steps: int = 10,
    ignore_occ: bool = True,
    ignore_oov: bool = True
) -> Tuple[List[float], List[float], List[float], float, float]:
    """
    Calculates "Average Accuracy".
    - Samples are split in `n_steps` threshold steps by their iou score between current and previous bbox;
    - Accuracy is calculated for each of these splits;
    - Final "Average Accuracy" is equal to mean of these values.
    - Empty splits are ignored

    Motivation:
        Accuracy is not enough to compare motion models in case when the objects are not moving most of the time.
        Example: Object is moving for 99% of the time (iou is 100% between the consecutive steps)
            and other 1% of time the object is moving very fast (iou is 5% between the consecutive steps).
            In that case the accuracy is equal to 0.99 * 100% + 0.01 * 5% = 99.05%
            but "Average Accuracy" is equal to (100% + 5%) / 2 = 52.5%.
        Both Accuracy and "Average Accuracy" should be used.

    Args:
        df: DataFrame
        n_steps: Number of split steps
        ignore_occ: Ignore occluded samples
        ignore_oov: Ignore out of view samples

    Returns:
        "Average Accuracy" metric.
    """
    # Preprocess
    if ignore_occ:
        df = df[df['occlusion'] == 0]
    if ignore_oov:
        df = df[df['out_of_view'] == 0]

    # Calculate metric
    step_size = 100 // n_steps
    steps = list(range(n_steps))
    thresholds = [s * step_size for s in steps]

    prior_scores, posterior_scores = [], []
    for i in steps:
        start = i * 10
        end = (i + 1) * 10

        df_t = df[(df['step_iou'] >= start) & (df['step_iou'] <= end)]
        if df_t.shape[0] == 0:
            continue

        prior_scores.append(df_t['prior_iou'].mean())
        posterior_scores.append(df_t['posterior_iou'].mean())

    prior_avg_iou = sum(prior_scores) / n_steps
    posterior_avg_iou = sum(posterior_scores) / n_steps

    return thresholds, prior_scores, posterior_scores, prior_avg_iou, posterior_avg_iou


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, experiment_path = pipeline.preprocess(cfg, name='average_accuracy', cls=ExtendedE2EGlobalConfig)
    cfg: ExtendedE2EGlobalConfig

    dataset = dataset_factory(
        name=cfg.dataset.name,
        path=cfg.dataset.fullpath,
        history_len=1,  # Not relevant
        future_len=1  # Not relevant
    )

    object_inferences_path = get_inference_path(experiment_path, cfg.end_to_end.filter.type)
    file_names = file_system.listdir(object_inferences_path, regex_filter='(.*?)[.]csv')
    file_paths = [os.path.join(object_inferences_path, fn) for fn in file_names]

    global_metrics = {
        'prior': {
            'global': [],
            'scenes': {},
            'categories': defaultdict(list)
        },
        'posterior': {
            'global': [],
            'scenes': {},
            'categories': defaultdict(list)
        }
    }
    inference_types = ['prior', 'posterior']

    for filepath in tqdm(file_paths, unit='file', desc='Calculating AverageAccuracy'):
        df = pd.read_csv(filepath)
        _, _, _, prior_avg_acc, posterior_avg_acc = average_accuracy(df)

        object_id = Path(filepath).stem
        for inference_type, avg_acc in zip(inference_types, [prior_avg_acc, posterior_avg_acc]):
            global_metrics[inference_type]['scenes'][object_id] = avg_acc
            category = dataset.get_object_category(object_id)
            global_metrics[inference_type]['categories'][category].append(avg_acc)
            global_metrics[inference_type]['global'].append(avg_acc)

    # Merge category stats
    for inference_type in inference_types:
        global_metrics[inference_type]['categories'] = {k: sum(v) / len(v) for k, v in global_metrics[inference_type]['categories'].items()}

        global_data = global_metrics[inference_type]['global']
        global_metrics[inference_type]['global'] = sum(global_data) / len(global_data)

    # Save evaluation
    output_metrics_path = os.path.join(object_inferences_path, 'average_accuracy_metrics.json')
    with open(output_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(global_metrics, f, indent=2)


if __name__ == '__main__':
    main()
