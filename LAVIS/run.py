

import subprocess
import random

def conver_list_str(input):

    result = "["

    for inner in input:

        if isinstance(inner, list):
            inner = [str(i) for i in inner]
            result += "[" + ",".join(inner) + "],"
        else:
            result += str(inner) + ","

    result = result[:-1]

    result += "]"

    return result

random.seed(0)
# python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml

num_layers = 24

total = 0

selected = []

num_samples = 60

for i in range(1, num_layers):
    selected.append([i-1, i])

others = []

for i in range(0, num_layers):
    for j in range(i + 2, num_layers):
        # print(i, j)

        others.append([i, j])

selected_others = random.sample(others, num_samples - len(selected))

selected = selected + selected_others


for s in selected:
    distilled_block_ids = list(set(range(num_layers)) - set(s))

    distilled_block_ids = conver_list_str(distilled_block_ids)

    job_id = conver_list_str(s)

    program = f"python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml --side_pretrained_weight 22-2048 --distilled_block_ids '{distilled_block_ids}' --job_id '{job_id}'"

    print(program)
    subprocess.call(program, shell=True)
