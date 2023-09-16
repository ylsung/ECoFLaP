"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import random
import time
import re
import os

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from functools import partial

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

    parser.add_argument(
        "--save_pruned_indices", action="store_true"
    )

    parser.add_argument(
        "--save_importance_measure", action="store_true"
    )

    parser.add_argument(
        "--importance_measure", type=str, default=None
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


def modify_distilled_clip(transformer, embed_dim, attn_dim, ffn_dim, layer_ids):
    # it is used to modify distilled model, so reinitializing the layers is fine
    match_string = rf".*blocks.({'|'.join(layer_ids)})*"

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(match_string, m_name):
            # print(m_name)

            module.attn.qkv = nn.Linear(embed_dim, attn_dim * 3, bias=False)
            if module.attn.q_bias is not None:
                module.attn.q_bias = nn.Parameter(torch.zeros(attn_dim))
                module.attn.v_bias = nn.Parameter(torch.zeros(attn_dim))
            module.attn.proj = nn.Linear(attn_dim, embed_dim)

            module.mlp.fc1 = nn.Linear(embed_dim, ffn_dim)
            module.mlp.fc2 = nn.Linear(ffn_dim, embed_dim)

    return transformer


def unstrct_generate_missing_mask(transformer, pruned_indices):

    for name, param in dict(transformer.state_dict()).items():
        if name not in pruned_indices:
            # don't prune them if the index are not existed in the pre-trained weight
            pruned_indices[name] = torch.ones_like(param, device=param.device, dtype=bool)

    return pruned_indices


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

    is_strct_pruning = False
    if args.distillation_init is not None:
        is_strct_pruning = "unstrct" in args.distillation_init

    if args.get_derivative_info:
        print("Setup for computing derivatice info")

        runner = RunnerBase(
            cfg=cfg, job_id=None, task=task, model=model, datasets=datasets
        )

        start = time.time()

        print("Start to compute derivatice info")
        derivative_info = runner.get_data_derivative(num_data=args.num_data, power=args.power, num_logits=args.num_logits)

        derivative_info = {k[7:]: v for k, v in derivative_info.items() if k.startswith("visual")} # filter out some info that is not for this transformer

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

        derivative_info = {k[7:]: v for k, v in derivative_info.items() if k.startswith("visual")} # filter out some info that is not for this transformer

        end = time.time()
        print(f"Finish computing activation info, using {end - start:.3f}s")
    else:
        derivative_info = None

    pruned_indices = None
    distilled_modify_func = None
    if args.pruned_indices is not None:
        pruned_indices = torch.load(args.pruned_indices)

        pruned_indices = pruned_indices["vit"]

        if is_strct_pruning:
            pruned_indices = unstrct_generate_missing_mask(model.visual, pruned_indices)
        else:
            p_device = pruned_indices[f"P_vit_ffn_38"].device
            
            # pruned_indices[f"P_vit_ffn_39"] = pruned_indices[f"P_vit_ffn_38"]
            
            ffn_dim = int(model.visual.embed_dim * model.visual.mlp_ratio)
            embed_dim = len(pruned_indices[f"P_vit_res"])

            num_heads = model.visual.num_heads

            pruned_indices[f"P_vit_ffn_39"] = torch.arange(ffn_dim).to(p_device)

            for i in range(num_heads):
                pruned_indices[f"P_vit_self_qk_39_{i}"] = torch.arange(embed_dim//num_heads).to(p_device)
                pruned_indices[f"P_vit_self_vo_39_{i}"] = torch.arange(embed_dim//num_heads).to(p_device)

            distilled_modify_func = partial(
                modify_distilled_clip, 
                embed_dim=embed_dim, 
                attn_dim=embed_dim,
                ffn_dim=ffn_dim, 
                layer_ids=['39'],
            )

    orig_total_size = sum(
        param.numel() for param in model.visual.parameters()
    )

    vit_importance_measure = None
    if args.importance_measure is not None:
        vit_importance_measure = torch.load(args.importance_measure)
        vit_importance_measure = vit_importance_measure["vit"]

    model.visual, vit_prune_indices, vit_importance_measure = vit_modify_with_weight_init(model.visual, args, False, "fp32", derivative_info, pruned_indices=pruned_indices, distilled_modify_func=distilled_modify_func, importance_measure=vit_importance_measure)

    if args.save_pruned_indices:

        saved_folder = "pruned_indices"
        os.makedirs(saved_folder, exist_ok=True)

        num_heads = model.visual.num_heads

        if is_strct_pruning:
            for k in list(vit_prune_indices.keys()):
                if "blocks.39" in k:
                    del vit_prune_indices[k]
                    print(f"{k} is deleted.")
        else:
            for k in list(vit_prune_indices.keys()):
                if "_39" in k:
                    del vit_prune_indices[k]
                    print(f"{k} is deleted.")

        pruned_indices = {
            "vit": vit_prune_indices,
        }

        torch.save(pruned_indices, os.path.join(saved_folder, job_id + ".pth"))

        print(os.path.join(saved_folder, job_id + ".pth"))

        exit()

    if args.save_importance_measure:
        saved_folder = "importance_measure"
        os.makedirs(saved_folder, exist_ok=True)

        importance_measure = {
            "vit": vit_importance_measure,
        }

        torch.save(importance_measure, os.path.join(saved_folder, job_id + ".pth"))

        print(os.path.join(saved_folder, job_id + ".pth"))

        exit()

    # model.t5_model, t5_prune_indices = t5_modify_with_weight_init(model.t5_model, args, derivative_info)

    # model.Qformer, model.t5_proj = qformer_pruning(
    #     model.Qformer, 
    #     model.t5_proj, 
    #     model.init_Qformer, 
    #     vit_prune_indices["P_vit_res"] if vit_prune_indices is not None else None, 
    #     t5_prune_indices["P_res"] if t5_prune_indices is not None else None
    # )

    if is_strct_pruning:
        distilled_total_size = sum(
            (param != 0).sum() for param in model.visual.parameters()
        )
    else:
        distilled_total_size = sum(
            param.numel() for param in model.visual.parameters()
        )

    # for name, param in model.t5_model.named_parameters():
    #     param.requires_grad = False

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    runner.orig_total_size = orig_total_size
    runner.distilled_total_size = distilled_total_size

    runner.evaluate(skip_reload=True)


if __name__ == "__main__":
    main()
