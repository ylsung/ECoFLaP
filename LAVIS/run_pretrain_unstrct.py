import subprocess
import random

import sys

GPU = sys.argv[1]
port = sys.argv[2]

method = "unstrct_mag_prune"

# for ratio in [0.5]:

    # ratios=f"1.0-1.0-{ratio}"
    # vit_pruned_indices=f"vit-{method}_{ratios}"

    # job_id = f"pretrain1-{vit_pruned_indices}"

    # program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run --nproc_per_node=1 --master_port {port} train.py"
    # f" --cfg-path lavis/projects/blip2/train/pretrain_stage1_cc3m.yaml"
    # f" --distillation_init '{method}' --vit_pruned_indices 'pruned_indices/{vit_pruned_indices}.pth'"
    # f" --vit_side_pretrained_weight 39-{ratios} --job_id '{job_id}'")

    # print(program)
    # subprocess.call(program, shell=True)



for ratio in [0.5]:

    # ratios=f"1.0-1.0-{ratio}"
    # vit_pruned_indices=f"vit-{method}_{ratios}"

    # job_id = f"pretrain1-{vit_pruned_indices}"

    # program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run --nproc_per_node=1 --master_port {port} train.py"
    # f" --cfg-path lavis/projects/blip2/train/pretrain_stage1_cc3m.yaml"
    # f" --distillation_init '{method}' --vit_pruned_indices 'pruned_indices/{vit_pruned_indices}.pth'"
    # f" --vit_side_pretrained_weight 39-{ratios} --job_id '{job_id}'")

    # print(program)
    # subprocess.call(program, shell=True)

    ratios=f"1.0-1.0-{ratio}"
    vit_pruned_indices=f"vit-{method}_{ratios}"
    t5_pruned_indices=f"t5-{method}_{ratios}"

    job_id = f"pretrain2_only-vit+t5-{method}_{ratios}"
    # pretrained = f"lavis/output/BLIP2/Pretrain_stage1/pretrain1-{vit_pruned_indices}/checkpoint_0.pth"
    pretrained = "/home/yilin/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth"

    program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run --nproc_per_node=1 --master_port {port} train.py"
    f" --cfg-path lavis/projects/blip2/train/pretrain_stage2_cc3m.yaml"
    f" --distillation_init '{method}' --vit_pruned_indices 'pruned_indices/{vit_pruned_indices}.pth'"
    f" --t5_pruned_indices 'pruned_indices/{t5_pruned_indices}.pth'"
    f" --vit_side_pretrained_weight 39-{ratios} --side_pretrained_weight 24-{ratios} --job_id '{job_id}'"
    f" --options pretrained={pretrained}")

    print(program)
    subprocess.call(program, shell=True)


    ratios = f"1.0-1.0-{ratio}"

    job_id = f"vit+t5-{method}_{ratios}"
    pretrained = f"lavis/output/BLIP2/Pretrain_stage2/pretrain2_only-{job_id}/checkpoint_0.pth"

    program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run --nproc_per_node=1 --master_port {port} evaluate.py"
    f" --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml"
    f" --distillation_init '{method}' --vit_pruned_indices 'pruned_indices/{vit_pruned_indices}.pth'"
    f" --t5_pruned_indices 'pruned_indices/{t5_pruned_indices}.pth'"
    f" --vit_side_pretrained_weight 39-{ratios} --side_pretrained_weight 24-{ratios} --job_id '{job_id}'"
    f" --options pretrained={pretrained}")

    print(program)
    subprocess.call(program, shell=True)
