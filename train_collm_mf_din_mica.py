"""
MiCA entry script.

Identical to train_collm_mf_din.py except it imports
minigpt4.models.minigpt4rec_mica so the MiCA registration
(mini_gpt4rec_mica) runs before registry is queried.
"""

import argparse
import os
import random
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Only one class is present in y_true")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="peft")
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

from minigpt4.datasets.builders import *
from minigpt4.models import *
import minigpt4.models.minigpt4rec_mica  # noqa: F401 - registers mini_gpt4rec_mica
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from torch.distributed.elastic.multiprocessing.errors import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", help="path to configuration file.")
    parser.add_argument("--options", nargs="+", help="override some settings")
    return parser.parse_args()


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "rec_runner_base"))
    return runner_cls


@record
def main():
    job_id = now()
    cfg = Config(parse_args())
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    data_name = list(datasets.keys())[0]
    try:
        data_dir = cfg.datasets_cfg.movie_ood.path
    except:
        data_dir = cfg.datasets_cfg.amazon_ood.path
    print("data dir:", data_dir)
    train_ = pd.read_pickle(data_dir + "train_ood2.pkl")
    valid_ = pd.read_pickle(data_dir + "valid_ood2.pkl")
    test_ = pd.read_pickle(data_dir + "test_ood2.pkl")
    user_num = max(train_.uid.max(), valid_.uid.max(), test_.uid.max()) + 1
    item_num = max(train_.iid.max(), valid_.iid.max(), test_.iid.max()) + 1
    cfg.model_cfg.rec_config.user_num = int(user_num)
    cfg.model_cfg.rec_config.item_num = int(item_num)
    cfg.pretty_print()
    model = task.build_model(cfg)
    runner = get_runner_class(cfg)(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    runner.train()


if __name__ == "__main__":
    main()
