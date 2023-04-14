

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

left_num_layers = 20

total = 0

selected = []

num_samples = 50


selected = [sorted(random.sample(range(1, num_layers), num_layers - left_num_layers)) for _ in range(num_samples)]


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
    distilled_block_ids = list(set(range(num_layers)) - set(s))

    assert len(distilled_block_ids) == left_num_layers
    distilled_block_ids = conver_list_str(distilled_block_ids)

    print(distilled_block_ids)

    job_id = "drop_4_" + conver_list_str(s)

    print(job_id)

    program = (f"python -m torch.distributed.run --nproc_per_node=2 evaluate.py"
    f" --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml --side_pretrained_weight '{left_num_layers}-1.0-1.0-1.0'"
    f" --distilled_block_ids '{distilled_block_ids}' --job_id '{job_id}'")

    print(program)
    subprocess.call(program, shell=True)