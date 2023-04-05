"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *

from lavis.compression.modify_model_with_weight_init import t5_modify_with_weight_init


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    parser.add_argument(
        "--side_pretrained_weight",
        type=str,
        default=None,
        help="The pre-trained config for the distilled transformer."
    )

    parser.add_argument(
        "--distillation_init",
        type=str,
        default="sum",
        help="Whether to init the distilled transformer."
    )

    parser.add_argument(
        "--distilled_block_ids",
        type=str,
        default=None,
        help="The layer assignment to merge the distilled transformer."
    )

    parser.add_argument(
        "--distilled_block_weights",
        type=str,
        default=None,
        help="The weight assignments to merge the distilled transformer."
    )

    parser.add_argument(
        "--modules_to_merge",
        type=str,
        default=".*",
        help="The type of modules to merge."
    )

    parser.add_argument(
        "--permute_before_merge",
        type=bool,
        default=False,
        help="Whether to permute the layers before merging (permute based on the first layer)"
    )

    parser.add_argument(
        "--permute_on_block_before_merge",
        type=bool,
        default=False,
        help="Whether to permute the layers before merging (permute independently based on blocks)"
    )

    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="The id of the Job"
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.

    args = parse_args()

    if args.job_id is not None:
        job_id = args.job_id
    else:
        job_id = now()

    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    model.t5_model = t5_modify_with_weight_init(model.t5_model, args)

    for name, param in model.t5_model.named_parameters():
        param.requires_grad = False

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.evaluate(skip_reload=True)


if __name__ == "__main__":
    main()
