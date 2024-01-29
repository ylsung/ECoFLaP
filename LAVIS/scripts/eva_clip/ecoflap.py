import subprocess
import random

import sys

GPU = sys.argv[1]
port = sys.argv[2]


method = "vit_wanda_pruner"
sparsity_ratio_granularity = "block"

score_method = "MEZO-GradOnly_sum"

ratio = 0.5
ratios = f"{ratio}-1.0-1.0"

max_sparsity_per_layer = f"{round(1.0 - ratio + 0.1, 1)}"
prunining_dataset_batch_size = 8

job_id = f"imgn-{method}_{ratios}_{score_method}{max_sparsity_per_layer}_{sparsity_ratio_granularity}_bs{prunining_dataset_batch_size}"

program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
f" --nproc_per_node=1 --master_port {port} evaluate_eva_clip.py"
f" --cfg-path lavis/projects/eva_clip/exp_imnet_zs_eval.yaml"
f" --pruning_method '{method}' --save_pruned_model"
f" --score_method {score_method}"
f" --sparsity_ratio_granularity {sparsity_ratio_granularity}"
f" --max_sparsity_per_layer {max_sparsity_per_layer}"
f" --prunining_dataset_batch_size {prunining_dataset_batch_size}"
f" --vit_prune_spec 40-{ratios} --job_id '{job_id}'")

print(program)
subprocess.call(program, shell=True)

method = "vit_wanda_pruner"

for task in ["exp_imnet_zs_eval"]:

    ratios = f"{ratio}-1.0-1.0"
    
    job_id = f"imgn-{method}_{ratios}_{score_method}{max_sparsity_per_layer}_{sparsity_ratio_granularity}_bs{prunining_dataset_batch_size}"

    vit_pruned_checkpoint = f"pruned_checkpoint/{job_id}.pth"

    program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
    f" --nproc_per_node=1 --master_port {port} evaluate_eva_clip.py"
    f" --cfg-path lavis/projects/eva_clip/{task}.yaml"
    f" --pruning_method '{method}'"
    f" --vit_pruned_checkpoint {vit_pruned_checkpoint}"
    f" --job_id '{job_id}'")

    print(program)
    subprocess.call(program, shell=True)
