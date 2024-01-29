import subprocess
import random

import sys

GPU = sys.argv[1]
port = sys.argv[2]


method = "blipt5_wanda_pruner"
sparsity_ratio_granularity = "block"

score_method = "olmezo-gradient_sum"

ratio = 0.5
ratios = f"{ratio}-1.0-1.0"

max_sparsity_per_layer = f"{round(1.0 - ratio + 0.1, 1)}"

for noise_eps in [1e-1, 1e-2, 1e-4]:

    job_id = f"cc3m-{method}_{ratios}_{score_method}{max_sparsity_per_layer}_{sparsity_ratio_granularity}_eps{noise_eps}"

    program = (f"CUDA_VISIBLE_DEVICES={GPU} python -m torch.distributed.run"
    f" --nproc_per_node=1 --master_port {port} evaluate_blip.py"
    f" --cfg-path lavis/projects/blip2/eval/cc_prefix_derivative_compute.yaml"
    f" --pruning_method '{method}' --save_pruned_model"
    f" --score_method {score_method}"
    f" --max_sparsity_per_layer {max_sparsity_per_layer}"
    f" --sparsity_ratio_granularity {sparsity_ratio_granularity}"
    f" --noise_eps {noise_eps}"
    f" --t5_prune_spec 24-{ratios} --vit_prune_spec 39-{ratios} --job_id '{job_id}'")

    print(program)
    subprocess.call(program, shell=True)

    method = "blipt5_wanda_pruner"

    for task in ["vqav2_zeroshot_flant5xl_eval", "gqa_zeroshot_flant5xl_eval", "okvqa_zeroshot_flant5xl_eval", "nocaps_flant5xl_eval", "ret_flickr_eval"]:

        ratios = f"{ratio}-1.0-1.0"
        
        job_id = f"cc3m-{method}_{ratios}_{score_method}{max_sparsity_per_layer}_{sparsity_ratio_granularity}_eps{noise_eps}"

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
