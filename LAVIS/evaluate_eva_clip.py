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

from lavis.compression import load_pruner


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
        "--num_data", type=int, default=128
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
    
    parser.add_argument(
        "--save_pruned_model", action="store_true"
    )
    
    parser.add_argument(
        "--vit_pruned_checkpoint",
        type=str,
        default=None,
        help="The pre-trained checkpoint for vit"
    )
    
    parser.add_argument(
        "--pruning_method", type=str,
    )
    
    parser.add_argument(
        "--vit_prune_spec",
        type=str,
        default=None,
        help="The pre-trained config for the distilled transformer."
    )
    
    parser.add_argument(
        "--sparsity_ratio_granularity",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--max_sparsity_per_layer", type=float, default=0.8
    )
    
    parser.add_argument(
        "--score_method",
        type=str,
        default="obd_avg",
    )
    
    parser.add_argument(
        "--num_data_first_stage", type=int, default=32
    )
    
    parser.add_argument(
        "--num_noise", default=1, type=int,
    )
    
    parser.add_argument(
        "--noise_eps", default=1e-3, type=float,
    )
    
    parser.add_argument(
        "--sparsity_dict",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--prune_per_model",
        action="store_true"
    )
    
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
    )
    
    parser.add_argument(
        "--is_global",
        action="store_true"
    )
    
    parser.add_argument(
        "--prunining_dataset_batch_size",
        type=int,
        default=1,
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

    orig_total_size = sum(
        param.numel() for param in model.visual.parameters()
    )
    
    sparsity_dict = None
    if args.vit_pruned_checkpoint is not None:
        print("Load vit pruned weight")
        prune_state_dict = torch.load(args.vit_pruned_checkpoint, map_location="cpu")
        
        model_prefix = None
        for candidate_prefix in ["visual.", "visual_encoder."]:
            if any(k.startswith(candidate_prefix) for k in prune_state_dict.keys()):
                model_prefix = candidate_prefix
                break
            
        assert model_prefix is not None
        
        prune_state_dict = {k: v for k, v in prune_state_dict.items() if k.startswith(model_prefix)}
        
        print(f"VIT checkpoint prefix: {model_prefix}")
        
        prune_state_dict = {k.replace(model_prefix, ""): v for k, v in prune_state_dict.items()}
        
        # prune_state_dict = {k.replace("visual.", ""): v for k, v in prune_state_dict.items()}
        original_checkpoints = model.visual.state_dict()
        original_checkpoints.update(prune_state_dict)
        prune_state_dict = original_checkpoints
        model.visual.load_state_dict(prune_state_dict)

    else:
        runner = RunnerBase(
            cfg=cfg, job_id=None, task=task, model=model, datasets=datasets
        )
        data_loader = runner.get_dataloader_for_importance_computation(
            num_data=args.num_data, power=args.power, batch_size=args.prunining_dataset_batch_size
        )
        
        config = {
            "prune_spec": args.vit_prune_spec,
            "importance_scores_cache": None,
            "keep_indices_cache": None,
            "is_strct_pruning": False,
            "is_global": args.is_global,
            "num_samples": args.num_data,
            "sparsity_ratio_granularity": args.sparsity_ratio_granularity,
            "max_sparsity_per_layer": args.max_sparsity_per_layer,
            "score_method": args.score_method,
            "num_data_first_stage": args.num_data_first_stage,
            "num_noise": args.num_noise,
            "noise_eps": args.noise_eps,
            "sparsity_dict": args.sparsity_dict,
            "prune_per_model": args.prune_per_model,
            "iteration": args.iteration,
        }

        # set up the classifier
        runner.task.before_evaluation(
            model=runner.unwrap_dist_model(runner.model).eval(),
            dataset=data_loader.dataset,
        )
        
        del model.text
        
        pruner = load_pruner(
            args.pruning_method, runner.unwrap_dist_model(runner.model).eval(), 
            data_loader, 
            cfg=config
        )
        model, sparsity_dict = pruner.prune()

    distilled_total_size = sum(
        (param != 0).float().sum() for param in model.visual.parameters()
    )
    
    print(distilled_total_size / orig_total_size * 100)
    
    if args.save_pruned_model:
        saved_folder = "pruned_checkpoint"
        os.makedirs(saved_folder, exist_ok=True)
        
        def filter_checkpoint(state_dict):
            def condition_to_keep(name):
                if "blocks.39" in name:
                    return False
                if "visual." not in name:
                    return False
                
                return True
                
            return {k: v for k, v in state_dict.items() if condition_to_keep(k)}
        
        torch.save(
            filter_checkpoint(model.state_dict()), 
            os.path.join(saved_folder, job_id + ".pth")
        )

        print(os.path.join(saved_folder, job_id + ".pth"))
        
        # save sparsity dict
        if sparsity_dict is not None and isinstance(sparsity_dict, dict):
            saved_folder = "sparsity_dict"
            os.makedirs(saved_folder, exist_ok=True)

            import yaml
            with open(os.path.join(saved_folder, job_id + ".yaml"), "w") as f:
                yaml.dump(sparsity_dict, f)

        exit()

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
