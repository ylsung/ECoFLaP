import subprocess
import random

import sys

GPU = sys.argv[1]
port = sys.argv[2]

for num_data in [256, 1024]:
    for ratio in [0.7]:

        ratios = f"1.0-1.0-{ratio}"

        job_id = f"t5-obs_prune{ratios}"

        program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run --nproc_per_node=1 --master_port {port} evaluate.py"
        f" --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml"
        f" --distillation_init 'obs_prune' --exact --to_one --get_derivative_info --num_data {num_data}"
        f" --side_pretrained_weight 24-{ratios} --job_id '{job_id}'")

        print(program)
        subprocess.call(program, shell=True)


for num_logits in [3, 10]:
    for ratio in [0.7]:

        ratios = f"1.0-1.0-{ratio}"

        job_id = f"t5-obs_prune{ratios}"

        program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run --nproc_per_node=1 --master_port {port} evaluate.py"
        f" --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml"
        f" --distillation_init 'obs_prune' --exact --to_one --get_derivative_info --num_logits {num_logits}"
        f" --side_pretrained_weight 24-{ratios} --job_id '{job_id}'")

        print(program)
        subprocess.call(program, shell=True)