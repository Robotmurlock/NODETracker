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
    analysis/*
"""
import os.path

CHECKPOINTS_DIRNAME = 'checkpoints'
INFERENCES_DIRNAME = 'inferences'
INFERENCE_VISUALIZATIONS_DIRNAME = 'visualizations'
TENSORBOARD_DIRNAME = 'tensorboard_logs'
CONFIGS_DIRNAME = 'configs'
ANALYSIS_DIRNAME = 'analysis'


def get_experiment_path(master_path: str, dataset_name: str, experiment_name: str) -> str:
    """
    Gets experiment location given 'master_path'

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
    Gets path to all checkpoints for given experiment

    Args:
        experiment_path: Path to experiment

    Returns:
        Checkpoints directory
    """
    return os.path.join(experiment_path, CHECKPOINTS_DIRNAME)


def get_checkpoint_path(experiment_path: str, checkpoint_filename: str) -> str:
    """
    Gets checkpoint full path.

    Args:
        experiment_path: Path to experiment
        checkpoint_filename: Checkpoint name

    Returns:
        Checkpoint full path
    """
    return os.path.join(get_checkpoints_dirpath(experiment_path), checkpoint_filename)


def get_config_path(experiment_path: str, config_name: str) -> str:
    """
    Gets configs full path.

    Args:
        experiment_path: Path to experiment
        config_name: Config name

    Returns:
        Config full path
    """
    return os.path.join(experiment_path, CONFIGS_DIRNAME, config_name)

def get_inference_fullname(model_type: str, dataset_name: str, split: str, experiment_name: str, inference_name: str) -> str:
    """
    Gets inference name by convention.

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
    Gets inference name convention.

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

def get_inference_video_path(inference_dirpath: str, scene_name: str, frame_range: str) -> str:
    """
    Gets inference visualization video path.

    Args:
        inference_dirpath: Inference directory path is acquired using `get_inference_path` function
        scene_name: Scene name
        frame_range: Frame range

    Returns:
        Path to visualization video
    """
    video_name = f'{scene_name}_{frame_range}.mp4'
    return os.path.join(inference_dirpath, INFERENCE_VISUALIZATIONS_DIRNAME, video_name)

def get_analysis_filepath(master_path: str, filename: str) -> str:
    """
    Gets analysis absolute filepath.

    Args:
        master_path: Global path (master path)
        filename: Filename

    Returns:
        Absolute path to filename
    """
    return os.path.join(master_path, ANALYSIS_DIRNAME, filename)
