import subprocess
import random

import sys

GPU = sys.argv[1]
port = sys.argv[2]

method = "global_obs_prune"

# for ratio in [0.7]:

#     ratios = f"1.0-1.0-{ratio}"

#     job_id = f"baseline_224"

#     program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run --nproc_per_node=1 --master_port {port} evaluate_eva_clip.py"
#     f" --cfg-path lavis/projects/eva_clip/exp_cifar100_zs_eval.yaml"
#     f" --job_id '{job_id}'")

#     print(program)
#     subprocess.call(program, shell=True)

for ratio in [0.7]:

    ratios = f"1.0-1.0-{ratio}"

    job_id = f"baseline_224"

    program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run --nproc_per_node=1 --master_port {port} evaluate_eva_clip.py"
    f" --cfg-path lavis/projects/eva_clip/exp_imnet_zs_eval.yaml"
    f" --job_id '{job_id}'")

    print(program)
    subprocess.call(program, shell=True)
