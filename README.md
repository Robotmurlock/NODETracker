# MoveSORT

Official code for paper [Beyond Kalman Filters: Deep Learning-Based Filters for Improved Object Tracking](https://arxiv.org/abs/2402.09865), 
i.e. **MoveSORT** tracker. Journal publication: [MVAA](https://link.springer.com/journal/138). Abstract:

```text
Traditional tracking-by-detection systems typically employ Kalman filters (KF) for state estimation. 
However, the KF requires domain-specific design choices and it is ill-suited to handling non-linear motion patterns. 
To address these limitations, we propose two innovative data-driven filtering methods. Our first method employs a Bayesian filter with a 
trainable motion model to predict an object's future location and combines its predictions with observations gained from an object detector to 
enhance bounding box prediction accuracy. Moreover, it dispenses with most domain-specific design choices characteristic of the KF. 
The second method, an end-to-end trainable filter, goes a step further by learning to correct detector errors, further minimizing the need 
for domain expertise. Additionally, we introduce a range of motion model architectures based on Recurrent Neural Networks, 
Neural Ordinary Differential Equations, and Conditional Neural Processes, that are combined with the proposed filtering methods. 
Our extensive evaluation across multiple datasets demonstrates that our proposed filters outperform the traditional KF in object tracking, 
especially in the case of non-linear motion patterns -- the use case our filters are best suited to. We also conduct noise robustness analysis 
of our filters with convincing positive results. We further propose a new cost function for associating observations with tracks. 
Our tracker, which incorporates this new association cost with our proposed filters, outperforms the conventional SORT method and other motion-based 
trackers in multi-object tracking according to multiple metrics on motion-rich DanceTrack and SportsMOT datasets. 
```

**Note**: Initial tracker name was `NODETracker` because of the `NODEFilter` motion model. However, MoveSORT
offers a generalized framework for deep-learning based Bayesian and end-to-end filters. 

## Filters

Supported filter architectures/methods:
- AR-RNN (AutoRegressive RNN, Bayesian filter)
- ACNP (Attentive Conditional Neural Processes, Bayesian filter)
- RNN-CNP (RNN Conditional Neural Processes, Bayesian filter)
- RNN-ODE (RNN + Neural ODE, Bayesian filter)
- RNNFilter (RNN-based end-to-end)
- RNNFilter (Neural ODE end-to-end)

## Datasets

Supported datasets: 
- [MOT17](https://motchallenge.net/), 
- [MOT20](https://motchallenge.net/), 
- [DanceTrack](https://github.com/DanceTrack/DanceTrack), 
- [SportsMOT](https://github.com/MCG-NJU/SportsMOT)

## Environment setup

### Python virtual environment

All required packages can be found in `requirements.txt`:
- `pip install -r requirements.txt`

### Docker

Alternative to python virtual environment is to just create a docker container. Build docker image:

```bash
docker-compose -f docker/docker-compose --env-file=.env build 
```

Run a container:

```bash
docker-compose -f docker/docker-compose --env-file=.env up -d
```

Container attach:

```bash
docker attach nodetracker-env
```

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
- `end_to_end`: Tracker inference configuration.

Every run of `tools` scripts saves the configuration in the logs directory.

### Training tools

The current set of `tools` scripts is:
- `python3 tools/run_train.py --config-path=<config_path> --config-name=<config_name>`: motion-model/filter training;
- `python3 tools/run_inference.py --config-path=<config_path> --config-name=<config_name>`: filtering inference and evaluation;
- `python3 tools/run_visualize.py --config-path=<config_path> --config-name=<config_name>`: filtering inference visualization.

Set of scripts for tracker evaluation:
- `python3 nodetracker/evaluation/end_to_end/tools/run_tracker_inference.py --config-path=<config_path> --config-name=<config_name>`, tracker inference
- `python3 nodetracker/evaluation/end_to_end/tools/run_tracker_postprocess.py --config-path=<config_path> --config-name=<config_name>`, tracker postprocess (interpolation)
- `python3 nodetracker/evaluation/end_to_end/tools/run_tracker_visualize_inference.py --config-path=<config_path> --config-name=<config_name>`, tracker inference visualization

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

## Citation

If you find this work useful, please consider to cite our paper:
```
@article{movesort,
  author = {Momir Adžemović and Predrag Tadić and Andrija Petrović and Mladen Nikolić},
  title = {Beyond Kalman filters: deep learning-based filters for improved object tracking},
  journal = {Machine Vision and Applications},
  volume = {36},
  number = {1},
  pages = {20},
  year = {2024},
  doi = {10.1007/s00138-024-01644-x},
  url = {https://doi.org/10.1007/s00138-024-01644-x},
  issn = {1432-1769}
}
```
