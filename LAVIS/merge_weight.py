import sys
import torch


base_weight = sys.argv[1]
other_weight = sys.argv[2]
output_weight = sys.argv[3]


base_weight = torch.load(base_weight, map_location="cpu")
other_weight = torch.load(other_weight, map_location="cpu")


for k in base_weight["model"].keys():
    if k in other_weight["model"]:
        base_weight["model"][k] = 0.5 * (base_weight["model"][k] + other_weight["model"][k])
        print("merge", k)
    else:
        print("extra", k)

torch.save(base_weight, output_weight)