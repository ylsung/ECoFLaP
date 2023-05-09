"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
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
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *

from lavis.compression.modify_model_with_weight_init import t5_modify_with_weight_init

from lavis.compression.modify_vit_with_weight_init import vit_modify_with_weight_init

from lavis.compression.modify_qformer_with_weight_init import pretrain_qformer_pruning, t5_proj_pruning


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
        "--save_pruned_indices", action="store_true"
    )

    parser.add_argument(
        "--vit_pruned_indices", type=str, default=None
    )

    parser.add_argument(
        "--t5_pruned_indices", type=str, default=None
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


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


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

    is_strct_pruning = False
    if args.distillation_init is not None:
        is_strct_pruning = "unstrct" in args.distillation_init

    derivative_info = None
    vit_derivative_info = None
    t5_derivative_info = None

    orig_total_size = sum(
        param.numel() for param in model.parameters()
    )

    vit_pruned_indices = None
    if args.vit_pruned_indices is not None:
        vit_pruned_indices = torch.load(args.vit_pruned_indices)
        vit_pruned_indices = vit_pruned_indices["vit"]
    
    model.visual_encoder, vit_pruned_indices = vit_modify_with_weight_init(model.visual_encoder, args, cfg.model_cfg.freeze_vit, cfg.model_cfg.vit_precision, vit_derivative_info, pruned_indices=vit_pruned_indices)

    t5_pruned_indices = None
    if getattr(model, "t5_model", None) is not None:
        t5_pruned_indices = None
        if args.t5_pruned_indices is not None:
            t5_pruned_indices = torch.load(args.t5_pruned_indices)
            t5_pruned_indices = t5_pruned_indices["t5"]
        model.t5_model, t5_pruned_indices = t5_modify_with_weight_init(model.t5_model, args, t5_derivative_info, pruned_indices=t5_pruned_indices)

        for name, param in model.t5_model.named_parameters():
            param.requires_grad = False

    if args.save_pruned_indices:

        saved_folder = "pruned_indices"
        os.makedirs(saved_folder, exist_ok=True)

        pruned_indices = {
            "t5": t5_pruned_indices,
            "vit": vit_pruned_indices,
        }

        torch.save(pruned_indices, os.path.join(saved_folder, job_id + ".pth"))

        print(os.path.join(saved_folder, job_id + ".pth"))

        exit()

    if is_strct_pruning:
        distilled_total_size = sum(
            (param != 0).float().sum() for param in model.parameters()
        )
    else:
        # only prune qformer for structural pruning
        model.Qformer = pretrain_qformer_pruning(
            model.Qformer, 
            model.init_Qformer, 
            vit_pruned_indices["P_vit_res"] if vit_pruned_indices is not None else None,
            len(model.tokenizer),
        )

        if getattr(model, "t5_proj", None) is not None:
            model.t5_proj = t5_proj_pruning(
                model.t5_proj, 
                t5_pruned_indices["P_res"] if t5_pruned_indices is not None else None,
            )

        distilled_total_size = sum(
            param.numel() for param in model.parameters()
        )

    print(f"{orig_total_size/10**9 :.3f} B, {distilled_total_size/10**9 :.3f} B")

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
