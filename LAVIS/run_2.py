

import subprocess
import random
import sys

GPU = sys.argv[1]
port = sys.argv[2]

for ratio in [0.8, 0.7, 0.6]:

    ratios = f"1.0-1.0-{ratio}"

    job_id = f"mag_prune{ratios}"

    program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run --nproc_per_node=1 --master_port {port} evaluate.py"
    f" --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml"
    f" --distillation_init 'mag_prune'"
    f" --side_pretrained_weight 24-{ratios} --job_id '{job_id}'")

    print(program)
    subprocess.call(program, shell=True)
