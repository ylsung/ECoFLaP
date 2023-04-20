

import subprocess
import random

import sys

GPU = sys.argv[1]
port = sys.argv[2]


for ratio in [0.7]:

    ratios = f"1.0-1.0-{ratio}"

    job_id = f"t5-zero_prune{ratios}"

    program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run --nproc_per_node=1 --master_port {port} evaluate.py"
    f" --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml"
    f" --distillation_init 'zero_prune' --exact --to_one"
    f" --side_pretrained_weight 24-{ratios} --job_id '{job_id}'")

    print(program)
    subprocess.call(program, shell=True)

for distill_merge_ratio in [0.25, 0.5, 0.75, 1.0]:
    for ratio in [0.7]:

        ratios = f"1.0-1.0-{ratio}"

        job_id = f"t5-zero_prune{ratios}+fusion{distill_merge_ratio}_to_one"

        program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run --nproc_per_node=1 --master_port {port} evaluate.py"
        f" --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml"
        f" --distillation_init 'zero_prune+fusion' --exact --to_one --distill_merge_ratio {distill_merge_ratio}"
        f" --side_pretrained_weight 24-{ratios} --job_id '{job_id}'")

        print(program)
        subprocess.call(program, shell=True)


# for distill_merge_ratio in [0.25, 0.5]:
#     for ratio in [0.7]:

#         ratios = f"1.0-1.0-{ratio}"

#         job_id = f"vit-mag_prune{ratios}+fusion{distill_merge_ratio}_imp"

#         program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run --nproc_per_node=1 --master_port {port} evaluate.py"
#         f" --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml"
#         f" --distillation_init 'mag_prune+fusion' --exact --importance --distill_merge_ratio {distill_merge_ratio}"
#         f" --vit_side_pretrained_weight 39-{ratios} --job_id '{job_id}'")

#         print(program)
#         subprocess.call(program, shell=True)

# for distill_merge_ratio in [0.25, 0.5]:
#     for ratio in [0.7]:

#         ratios = f"1.0-1.0-{ratio}"

#         job_id = f"vit-mag_prune{ratios}+fusion{distill_merge_ratio}_to_one_imp"

#         program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run --nproc_per_node=1 --master_port {port} evaluate.py"
#         f" --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml"
#         f" --distillation_init 'mag_prune+fusion' --exact --to_one --importance --distill_merge_ratio {distill_merge_ratio}"
#         f" --vit_side_pretrained_weight 39-{ratios} --job_id '{job_id}'")

#         print(program)
#         subprocess.call(program, shell=True)