import subprocess
import random

import sys

GPU = sys.argv[1]
port = sys.argv[2]


method = "t5_wanda_pruner"
sparsity_ratio_granularity = "block"

score_method = "MEZO-GradOnly_avg"

ratio = 0.5
ratios = f"{ratio}-1.0-1.0"

max_sparsity_per_layer = f"{round(1.0 - ratio + 0.1, 1)}"
prunining_dataset_batch_size = 8

job_id = f"cc3m-{method}_{ratios}_{score_method}{max_sparsity_per_layer}_{sparsity_ratio_granularity}_bs{prunining_dataset_batch_size}"

program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
f" --nproc_per_node=1 --master_port {port} evaluate_t5.py"
f" --cfg-path lavis/projects/blip2/eval/c4_prefix_derivative_compute.yaml"
f" --pruning_method '{method}' --save_pruned_model"
f" --score_method {score_method}"
f" --sparsity_ratio_granularity {sparsity_ratio_granularity}"
f" --max_sparsity_per_layer {max_sparsity_per_layer}"
f" --prunining_dataset_batch_size {prunining_dataset_batch_size}"
f" --t5_prune_spec 24-{ratios} --job_id '{job_id}'")

print(program)
subprocess.call(program, shell=True)
