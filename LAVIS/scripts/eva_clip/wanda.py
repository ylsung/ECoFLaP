import subprocess
import random

import sys

GPU = sys.argv[1]
port = sys.argv[2]


method = "vit_wanda_pruner"

ratio = 0.5
ratios = f"{ratio}-1.0-1.0"

job_id = f"imgn-{method}_{ratios}"

program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
f" --nproc_per_node=1 --master_port {port} evaluate_eva_clip.py"
f" --cfg-path lavis/projects/eva_clip/exp_imnet_zs_eval.yaml"
f" --pruning_method '{method}' --save_pruned_model"
f" --vit_prune_spec 40-{ratios} --job_id '{job_id}'")

print(program)
subprocess.call(program, shell=True)

method = "vit_wanda_pruner"

for task in ["exp_imnet_zs_eval"]:

    ratios = f"{ratio}-1.0-1.0"
    
    job_id = f"imgn-{method}_{ratios}"

    vit_pruned_checkpoint = f"pruned_checkpoint/{job_id}.pth"

    program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
    f" --nproc_per_node=1 --master_port {port} evaluate_eva_clip.py"
    f" --cfg-path lavis/projects/eva_clip/{task}.yaml"
    f" --pruning_method '{method}'"
    f" --vit_pruned_checkpoint {vit_pruned_checkpoint}"
    f" --job_id '{job_id}'")

    print(program)
    subprocess.call(program, shell=True)
