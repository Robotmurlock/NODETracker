"""
Collects experiment configs and evaluation metrics.
"""
import argparse
import logging
import os
import shutil
from pathlib import Path
import traceback

from tqdm import tqdm

from nodetracker.common import conventions
from nodetracker.utils.logging import configure_logging

logger = logging.getLogger('CollectExperimentData')

def main(args: argparse.Namespace) -> None:
    input_path = args.input_path
    output_path = args.output_path
    dataset_metrics_filename = 'dataset_metrics.json'

    experiment_directories = os.listdir(input_path)
    for experiment_name in tqdm(experiment_directories, unit='experiment'):
        logger.info(f'Found experiment "{experiment_name}".')
        experiment_path = os.path.join(input_path, experiment_name)
        # noinspection PyBroadException
        try:

            train_config_path = conventions.get_config_path(experiment_path, 'train.yaml')
            inferences_path = conventions.get_inferences_dirpath(experiment_path)
            inference_directories = os.listdir(inferences_path)

            for inference_name in inference_directories:
                logger.info(f'Found experiment inference "{inference_name}".')
                inference_dirpath = os.path.join(inferences_path, inference_name)
                result_path = os.path.join(output_path, experiment_name, inference_name)
                Path(result_path).mkdir(parents=True, exist_ok=True)

                inference_config_filepath = conventions.get_inference_config_path(inference_dirpath)
                metrics_filepath = os.path.join(inference_dirpath, dataset_metrics_filename)
                shutil.copy(metrics_filepath, os.path.join(result_path, dataset_metrics_filename))
                shutil.copy(train_config_path, os.path.join(result_path, 'train.yaml'))
                shutil.copy(inference_config_filepath, os.path.join(result_path, 'inference.yaml'))
        except:
            logger.error(f'Failed to collect "{experiment_name}"\n{traceback.format_exc()}')


def parse_configs() -> argparse.Namespace:
    """
    Returns:
        Parsed configs
    """
    parser = argparse.ArgumentParser(description='Script Arguments Parser')
    parser.add_argument('--input-path', type=str, required=True, help='Path to dataset outputs.')
    parser.add_argument('--output-path', type=str, required=False, default='collected_experiments', help='Path to script output')
    return parser.parse_args()


if __name__ == '__main__':
    configure_logging(logging.DEBUG)
    main(parse_configs())
