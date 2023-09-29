"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import random
import time
import os

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
from lavis.compression.woodfisher.woodfisher import WoodFisher

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
        "--save_pruned_indices", action="store_true"
    )

    parser.add_argument(
        "--vit_pruned_indices", type=str, default=None
    )

    parser.add_argument(
        "--t5_pruned_indices", type=str, default=None
    )

    parser.add_argument(
        "--save_importance_measure", action="store_true"
    )

    parser.add_argument(
        "--vit_importance_measure", type=str, default=None
    )

    parser.add_argument(
        "--t5_importance_measure", type=str, default=None
    )
    
    parser.add_argument(
        "--t5_pruned_checkpoint", type=str, default=None
    )
    
    parser.add_argument(
        "--vit_pruned_checkpoint", type=str, default=None
    )
    
    parser.add_argument(
        "--t5_prune_spec", type=str, default=None
    )
    
    parser.add_argument(
        "--vit_prune_spec", type=str, default=None
    )

    parser.add_argument(
        "--vision_weight", type=float, default=0.0
    )

    parser.add_argument(
        "--save_final_activations", action="store_true"
    )
    
    parser.add_argument(
        "--pruning_method", type=str, default="blipt5_wanda_pruner",
    )
    
    parser.add_argument(
        "--save_pruned_model", action="store_true"
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
        "--sparsity_dict",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--prune_per_model",
        action="store_true"
    )
    
    parser.add_argument(
        "--is_global",
        action="store_true"
    )
    
    parser.add_argument(
        "--iteration",
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


def get_final_activations(args, cfg, task, model, datasets):
    runner = RunnerBase(
        cfg=cfg, job_id=None, task=task, model=model, datasets=datasets
    )
    start = time.time()

    print("Start to get final activation")
    outputs = runner.get_last_activations(num_data=args.num_data, power=args.power)

    end = time.time()
    print(f"Finish get final activation, using {end - start:.3f}s")

    return outputs


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
        param.numel() for param in model.parameters()
    )
    
    if args.t5_pruned_checkpoint is not None and getattr(model, "t5_model", None) is not None:
        print("Load t5 pruned weight")
        prune_state_dict = torch.load(args.t5_pruned_checkpoint, map_location="cpu")
        
        prune_state_dict = {k: v for k, v in prune_state_dict.items() if k.startswith("t5_model")}
        
        prune_state_dict = {k.replace("t5_model.", ""): v for k, v in prune_state_dict.items()}
        model.t5_model.load_state_dict(prune_state_dict)
        
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
        
        # additional_keys = [
        #     "norm.weight", "norm.bias", "head.weight", "head.bias"
        # ]
        
        original_state_dict = model.visual_encoder.state_dict()
        
        for k, v in prune_state_dict.items():
            if k in original_state_dict:
                original_state_dict[k] = v
                
        prune_state_dict = original_state_dict
        
        from lavis.models.eva_vit import interpolate_pos_embed
        
        interpolate_pos_embed(model.visual_encoder, prune_state_dict)
        # for additional_key in additional_keys:
        #     del prune_state_dict[additional_key]

        model.visual_encoder.load_state_dict(prune_state_dict)
        
    runner = RunnerBase(
        cfg=cfg, job_id=None, task=task, model=model, datasets=datasets
    )
    data_loader = runner.get_dataloader_for_importance_computation(num_data=args.num_data, power=args.power)

    config = {
        "t5_prune_spec": args.t5_prune_spec if args.t5_pruned_checkpoint is None else None,
        "vit_prune_spec": args.vit_prune_spec if args.vit_pruned_checkpoint is None else None,
        "t5_pruning_method": "none",
        "vit_pruning_method": "none",
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
        "sparsity_dict": args.sparsity_dict,
        "prune_per_model": args.prune_per_model,
        "iteration": args.iteration,
    }
    
    pruner = load_pruner(
        args.pruning_method, runner.unwrap_dist_model(runner.model).eval(), 
        data_loader, 
        cfg=config
    )
    
    start = time.time()
    model, sparsity_dict = pruner.prune()

    # model, _ = pruner.prune()

    distilled_total_size = sum(
        (param != 0).float().sum() for param in model.parameters()
    )
    
    print(distilled_total_size / orig_total_size * 100)
    
    if args.save_pruned_model:
        saved_folder = "pruned_checkpoint"
        os.makedirs(saved_folder, exist_ok=True)
        
        torch.save(
            model.state_dict(), 
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
                
        
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        
        processing_time = time.time() - start
        
        training_dict = {
            "memory": peak_memory,
            "time": processing_time
        }
        
        saved_folder = "training_statistics"
        os.makedirs(saved_folder, exist_ok=True)
        
        import yaml
        with open(os.path.join(saved_folder, job_id + ".yaml"), "w") as f:
            yaml.dump(training_dict, f)
            
        
        saved_folder = "importance_scores"
        os.makedirs(saved_folder, exist_ok=True)
        
        torch.save(
            {k: v.importance_score for k, v in model.named_parameters() if getattr(v, "importance_score", None) is not None}, 
            os.path.join(saved_folder, job_id + ".pth")
        )

        
        exit()

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    runner.orig_total_size = orig_total_size
    runner.distilled_total_size = distilled_total_size

    runner.evaluate(skip_reload=True)


if __name__ == "__main__":
    main()