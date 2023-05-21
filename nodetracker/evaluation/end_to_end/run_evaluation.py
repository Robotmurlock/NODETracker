import hydra
from nodetracker.common.project import CONFIGS_PATH
from nodetracker.utils import pipeline
from nodetracker.evaluation.end_to_end.config import ExtendedE2EGlobalConfig
from omegaconf import DictConfig
from nodetracker.datasets.factory import dataset_factory


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
def main(cfg: DictConfig):
    cfg, _ = pipeline.preprocess(cfg, name='e2e_evaluation', cls=ExtendedE2EGlobalConfig)
    cfg: ExtendedE2EGlobalConfig

    dataset = dataset_factory(
        name=cfg.dataset.name,
        path=cfg.dataset.fullpath,
        sequence_list=cfg.dataset.split_index[cfg.eval.split],
        history_len=1,  # Not relevant
        future_len=1  # not relevant
    )




if __name__ == '__main__':
    main()
