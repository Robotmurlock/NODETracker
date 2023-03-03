# NODETracker

## Datasets

Overview of all datasets.

### MOT20

**MOT20** is multi-object-tracking pedestrian tracking dataset. All scenes have static camera views. 
Detailed dataset information can be found in the original [paper](https://arxiv.org/pdf/2003.09003.pdf).

Dataset can be acquired [here](https://motchallenge.net/data/MOT20/) or
by running the bash script `nodetracker/utility_tools/run_download_and_setup_mot20.sh` (recommended).

## Environment setup

All required packages can be found in `requirements.txt`:
- `pip install -r requirements.txt`

## Scripts

This section explains shortly all important steps in order to configure and run scripts.
All main scripts (training, inference, visualization) can be found in `tools` directory.

### Configuration

All `tools` scripts use the same configuration (global configuration).
Configuration components:
- `resources`: Resource configuration (CPU cores, use CUDA/CPU, ...);
- `dataset`: Dataset configuration (path, train, val, ...);
- `transform`: Data transform configuration (preprocess and postprocess);
- `model`: Model configuration (Model and hyperparameter selection);
- `train`: Training configuration (max epochs, batch size, ...);
- `eval`: Evaluation (with inference) configuration;
- `visualize`: Inference visualization configuration.

Every run of `tools` scripts saves the configuration in the logs directory.

### Script tools

The current set of `tools` scripts is:
- `run_train`: training;
- `run_inference`: inference and evaluation;
- `run_visualize`: inference visualization.

### Generated data structure

All generated data is stored in `master_path` path which is configurable. Structure:

```
{dataset_name}/
    {experiment_name}/
        checkpoints/*
        configs/*
            train.yaml (sacuvane konfiguracije za pokrenute skripte)
            inference.yaml
            visualize.yaml
        inferences/*
            {inference_name}/*
                [visualize]/*   
                ...
        tensorboard_logs/*
```

### Model Zoo

Configurations for all models and links to model checkpoints can be found in `models` section.

## Papers

Published versions of paper can be found in `papers` section.
