"""
Dataset, model and inference file system structure

Pipeline data structure:
{master_path}/
    {dataset_name}/
        {experiment_name}/
            checkpoints/
                {checkpoint_name_1}
                {checkpoint_name_2}
                ...
            configs/
                [train.yaml]
                [inference.yaml]
                [visualize.yaml]
                ...
            inferences/
                {inference_name_1}/*
                {inference_name_2}/*
                ...
            tensorboard_logs/*
"""
import os.path

CHECKPOINTS_DIRNAME = 'checkpoints'
INFERENCES_DIRNAME = 'inferences'
TENSORBOARD_DIRNAME = 'tensorboard_logs'
CONFIGS_DIRNAME = 'configs'


def get_experiment_path(master_path: str, dataset_name: str, experiment_name: str) -> str:
    """
    Get experiment location given 'master_path'

    Args:
        master_path: Global path (master path)
        dataset_name: Dataset name
        experiment_name: Model experiment name

    Returns:
        Experiment absolute path
    """
    return os.path.join(master_path, dataset_name, experiment_name)


def get_checkpoints_dirpath(experiment_path: str) -> str:
    """
    Get path to all checkpoints for given experiment

    Args:
        experiment_path: Path to experiment

    Returns:
        Checkpoints directory
    """
    return os.path.join(experiment_path, CHECKPOINTS_DIRNAME)


def get_checkpoint_path(experiment_path: str, checkpoint_filename: str) -> str:
    """
    Get checkpoint full path.

    Args:
        experiment_path: Path to experiment
        checkpoint_filename: Checkpoint name

    Returns:
        Checkpoint full path
    """
    return os.path.join(get_checkpoints_dirpath(experiment_path), checkpoint_filename)


def get_config_path(experiment_path: str, config_name: str) -> str:
    """
    Get configs full path.

    Args:
        experiment_path: Path to experiment
        config_name: Config name

    Returns:
        Config full path
    """
    return os.path.join(experiment_path, CONFIGS_DIRNAME, config_name)

def get_inference_fullname(model_type: str, dataset_name: str, split: str, experiment_name: str, inference_name: str) -> str:
    """
    Git inference name by convention.

    Args:
        model_type: Model type (architecture)
        dataset_name: Dataset name
        split: Split (train, val, test)
        experiment_name: Model experiment name
        inference_name: Inference name

    Returns:
        Inference fullname (unique)
    """
    for inf_name_component in [model_type, dataset_name, split, experiment_name, inference_name]:
        if '_' in inf_name_component:
            raise ValueError(f'Found "_" in {inf_name_component}. Please use some other character!')

    return f'{model_type}_{dataset_name}_{split}_{experiment_name}_{inference_name}'

def get_inference_path(experiment_path: str, model_type: str, dataset_name: str, split: str, experiment_name: str, inference_name: str) -> str:
    """
    Inference name convention

    Args:
        experiment_path: Path to experiment
        model_type: Model type (architecture)
        dataset_name: Dataset name
        split: Split (train, val, test)
        experiment_name: Model experiment name
        inference_name: Inference name

    Returns:
        Inference fullname (unique)
    """
    inf_fullname = get_inference_fullname(model_type, dataset_name, split, experiment_name, inference_name)
    return os.path.join(experiment_path, INFERENCES_DIRNAME, inf_fullname)
