

import subprocess
import random

for ratio in [0.9, 0.8, 0.7, 0.6]:

    job_id = f"mag_prune{ratio}"

    program = (f"python -m torch.distributed.run --nproc_per_node=3 evaluate.py"
    f" --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml"
    f" --distillation_init 'mag_prune'"
    f" --side_pretrained_weight 24-{ratio} --job_id '{job_id}'")

    print(program)
    subprocess.call(program, shell=True)
