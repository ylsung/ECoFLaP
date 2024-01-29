import subprocess
import random

import sys

GPU = sys.argv[1]
port = sys.argv[2]


method = "blipt5_sparsegpt_pruner"

ratio = 0.4
ratios = f"{ratio}-1.0-1.0"

job_id = f"cc3m-{method}_{ratios}_olmezo-gradient_sum0.7_block"

sparsity_dict = f"cc3m-blipt5_wanda_pruner_0.4-1.0-1.0_olmezo-gradient_sum0.7_block"

program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
f" --nproc_per_node=1 --master_port {port} evaluate_blip.py"
f" --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute.yaml"
f" --pruning_method '{method}' --save_pruned_model"
f" --sparsity_dict sparsity_dict/{sparsity_dict}.yaml"
f" --t5_prune_spec 24-{ratios} --vit_prune_spec 39-{ratios} --job_id '{job_id}'")

print(program)
subprocess.call(program, shell=True)

for task in ["vqav2_zeroshot_flant5xl_eval", "gqa_zeroshot_flant5xl_eval", "okvqa_zeroshot_flant5xl_eval", "nocaps_flant5xl_eval", "ret_flickr_eval"]:

    ratios = f"{ratio}-1.0-1.0"
    
    job_id = f"cc3m-{method}_{ratios}_olmezo-gradient_sum0.7_block"

    vit_pruned_checkpoint = f"pruned_checkpoint/{job_id}.pth"
    t5_pruned_checkpoint = f"pruned_checkpoint/{job_id}.pth"

    program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
    f" --nproc_per_node=1 --master_port {port} evaluate_blip.py"
    f" --cfg-path lavis/projects/blip2/eval/{task}.yaml"
    f" --pruning_method '{method}'"
    f" --t5_pruned_checkpoint {t5_pruned_checkpoint}"
    f" --vit_pruned_checkpoint {vit_pruned_checkpoint}"
    f" --job_id '{job_id}'")

    print(program)
    subprocess.call(program, shell=True)
