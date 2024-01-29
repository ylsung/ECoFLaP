import subprocess
import random

import sys

GPU = sys.argv[1]
port = sys.argv[2]


method = "t5_wanda_pruner"

ratio = 0.5
ratios = f"{ratio}-1.0-1.0"

job_id = f"cc3m-{method}_{ratios}"

program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
f" --nproc_per_node=1 --master_port {port} evaluate_t5.py"
f" --cfg-path lavis/projects/blip2/eval/c4_prefix_derivative_compute.yaml"
f" --pruning_method '{method}' --save_pruned_model"
f" --t5_prune_spec 24-{ratios} --job_id '{job_id}'")

print(program)
subprocess.call(program, shell=True)
