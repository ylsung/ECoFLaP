"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import random
import time

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

from lavis.compression.modify_vit_with_weight_init import vit_modify_with_weight_init

from lavis.compression.modify_qformer_with_weight_init import qformer_pruning


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
        "--vit_side_pretrained_weight",
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
        action="store_true",
        default=False,
        help="Whether to permute the layers before merging (permute based on the first layer)"
    )

    parser.add_argument(
        "--permute_on_block_before_merge",
        action="store_true",
        default=False,
        help="Whether to permute the layers before merging (permute independently based on blocks)"
    )

    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="The id of the Job"
    )

    parser.add_argument(
        "--vit_ffn_ratio", type=float, default=1.0
    )

    parser.add_argument(
        "--distilled_merge_ratio", type=float, default=0.5
    )

    parser.add_argument(
        "--exact", action="store_true"
    )

    parser.add_argument(
        "--normalization", action="store_true"
    )

    parser.add_argument(
        "--metric", type=str, default="dot"
    )

    parser.add_argument(
        "--distill_merge_ratio", type=float, default=0.5
    )

    parser.add_argument(
        "--to_one", action="store_true"
    )

    parser.add_argument(
        "--importance", action="store_true"
    )

    parser.add_argument(
        "--num_data", type=int, default=64
    )

    parser.add_argument(
        "--power", type=int, default=2
    )

    parser.add_argument(
        "--num_logits", type=int, default=1
    )

    parser.add_argument(
        "--get_derivative_info", action="store_true"
    )

    parser.add_argument(
        "--get_activation_info", action="store_true"
    )

    parser.add_argument(
        "--use_input_activation", action="store_true"
    )

    parser.add_argument(
        "--pruned_indices", type=str, default=None
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

    args = parse_args()

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
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

    if args.get_derivative_info:
        print("Setup for computing derivatice info")

        runner = RunnerBase(
            cfg=cfg, job_id=None, task=task, model=model, datasets=datasets
        )

        start = time.time()

        print("Start to compute derivatice info")
        derivative_info = runner.get_data_derivative(num_data=args.num_data, power=args.power, num_logits=args.num_logits)

        end = time.time()
        print(f"Finish computing derivatice info, using {end - start:.3f}s")
        # for n, p in derivative_info.items():
        #     print(n, p.shape)

    elif args.get_activation_info:
        print("Setup for computing activation info")

        runner = RunnerBase(
            cfg=cfg, job_id=None, task=task, model=model, datasets=datasets
        )

        start = time.time()

        print("Start to compute activation info")
        input_representations, output_representations = runner.get_activations(num_data=args.num_data, power=args.power)

        derivative_info = runner.convert_activation_to_importance(input_representations, output_representations, args.use_input_activation)

        end = time.time()
        print(f"Finish computing activation info, using {end - start:.3f}s")
    else:
        derivative_info = None

    pruned_indices = None
    if args.pruned_indices is not None:
        pruned_indices = torch.load(args.pruned_indices)

        pruned_indices = pruned_indices["vit"]

        p_device = pruned_indices[f"P_vit_ffn_38"].device
        
        pruned_indices[f"P_vit_ffn_39"] = pruned_indices[f"P_vit_ffn_38"]

    # orig_total_size = sum(
    #     param.numel() for param in model.parameters()
    # )

    model.visual, vit_prune_indices = vit_modify_with_weight_init(model.visual, args, False, "fp32", None, pruned_indices=pruned_indices)

    # model.t5_model, t5_prune_indices = t5_modify_with_weight_init(model.t5_model, args, derivative_info)

    # model.Qformer, model.t5_proj = qformer_pruning(
    #     model.Qformer, 
    #     model.t5_proj, 
    #     model.init_Qformer, 
    #     vit_prune_indices["P_vit_res"] if vit_prune_indices is not None else None, 
    #     t5_prune_indices["P_res"] if t5_prune_indices is not None else None
    # )

    # distilled_total_size = sum(
    #     param.numel() for param in model.parameters()
    # )

    # for name, param in model.t5_model.named_parameters():
    #     param.requires_grad = False

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    # runner.orig_total_size = orig_total_size
    # runner.distilled_total_size = distilled_total_size

    runner.evaluate(skip_reload=True)


if __name__ == "__main__":
    main()
