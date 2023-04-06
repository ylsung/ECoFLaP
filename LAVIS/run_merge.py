

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

print(len(selected))


def get_merge_block_ids(selected_idx, num_layers):
    selected_idx = sorted(selected_idx) # make sure it is sorted
    distilled_block_ids = list(range(num_layers))

    def get_idx(s, blocks):
        for i in range(len(blocks)):
            if blocks[i] == s:
                return i

        raise ValueError("The merge layer is not in the pre-trained model")

    for selected_id in selected_idx:
        idx = get_idx(selected_id, distilled_block_ids)

        if isinstance(distilled_block_ids[idx - 1], list):
            distilled_block_ids[idx - 1].append(distilled_block_ids[idx])
        else:
            distilled_block_ids[idx - 1] = [distilled_block_ids[idx - 1], distilled_block_ids[idx]]

        distilled_block_ids.pop(idx)

    return distilled_block_ids


for s in selected:
    if 0 in s:
        # in merge case, we will never merge 0 to 1, because it is the same as merge 1 to 0
        continue
    distilled_block_ids = get_merge_block_ids(s, num_layers)

    distilled_block_ids = conver_list_str(distilled_block_ids)

    print(distilled_block_ids)

    job_id = "merge_ffnLN_" + conver_list_str(s)

    print(job_id)

    program = (f"python -m torch.distributed.run --nproc_per_node=4 evaluate.py"
    f" --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml --side_pretrained_weight 22-2048"
    f" --distilled_block_ids '{distilled_block_ids}' --job_id '{job_id}' --modules_to_merge '.*layer_norm.*|.*DenseReluDense.*'"
    f" --permute_on_block_before_merge")

    print(program)
    subprocess.call(program, shell=True)


for s in selected:
    if 0 in s:
        # in merge case, we will never merge 0 to 1, because it is the same as merge 1 to 0
        continue
    distilled_block_ids = get_merge_block_ids(s, num_layers)

    distilled_block_ids = conver_list_str(distilled_block_ids)

    print(distilled_block_ids)

    job_id = "merge_all_" + conver_list_str(s)

    print(job_id)

    program = (f"python -m torch.distributed.run --nproc_per_node=4 evaluate.py"
    f" --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml --side_pretrained_weight 22-2048"
    f" --distilled_block_ids '{distilled_block_ids}' --job_id '{job_id}' --modules_to_merge '.*|.*'"
    f" --permute_on_block_before_merge")

    print(program)
    subprocess.call(program, shell=True)
