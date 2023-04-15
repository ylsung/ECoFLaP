import torch
import torch.nn as nn
from copy import deepcopy

from lavis.compression.weight_matching import permutation_spec_from_axes_to_perm, weight_matching, apply_permutation, ot_weight_fusion, apply_permutation_by_matrix 

total = 100 # number of trials
correct = 0 # number of successful matching

for iteration in range(total):
    d1 = 500
    d2 = 300

    x = torch.randn(5, d1)

    # create the weights for two models

    # The model has residual connections and looks like (this architecture is common in Transformers)
    #
    #        fc1        fc2        fc3        fc4
    #    x ------> z1 ------> z2 ------> z3 ------> o 
    #     |                   ^ |                   ^        
    #     |  _________________| | __________________|
    #      residual connection    residual connection
    # 

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc0 = nn.Linear(d1, d1)
            self.fc1 = nn.Linear(d1, d2)
            self.fc2 = nn.Linear(d2, d1)
            self.fc3 = nn.Linear(d1, d2)
            self.fc4 = nn.Linear(d2, d1)
            self.fc5 = nn.Linear(d1, d2)
        
        def forward(self, x):
            out_0 = self.fc0(x)
            out_2 = self.fc2(self.fc1(out_0)) + out_0
            out_4 = self.fc4(self.fc3(out_2)) + out_2

            return self.fc5(out_4)

    model_a = Model()
    model_b = Model()

    params_a = model_a.state_dict()
    params_b = deepcopy(params_a)

    # create the hand-crafted permutation for the weights that are connected by residual connections
    res_p = torch.randperm(d1)

    last_p = None

    # create the hand-crafted permutation for the the other weights 
    # and use hand-crafted permutations to permute the params_b
    for i in range(0, 6):
        n = f"fc{i}"

        if i == 5:
            # last layer
            params_b[n + ".weight"] = params_b[n + ".weight"][:, last_p]
        else:
            if i % 2 == 0: # fc0, fc2, fc4
                p = res_p
            else: # fc1, fc3, fc5
                p = torch.randperm(d2)

            # permute the weight and bias
            if last_p is not None:
                params_b[n + ".weight"] = params_b[n + ".weight"][p, :][:, last_p]
            else:
                params_b[n + ".weight"] = params_b[n + ".weight"][p, :]

            params_b[n + ".bias"] = params_b[n + ".bias"][p]

            last_p = p


    model_b.load_state_dict(params_b)

    print(torch.all(torch.isclose(model_a(x), model_b(x))))


    # determine the spec
    ps = permutation_spec_from_axes_to_perm({
        "fc0.weight": ("P_res", None),
        "fc1.weight": ("P_1", "P_res"),
        "fc2.weight": ("P_res", "P_1"),
        "fc3.weight": ("P_2", "P_res"),
        "fc4.weight": ("P_res", "P_2"),
        "fc5.weight": (None, "P_res"),

        "fc0.bias": ("P_res",),
        "fc1.bias": ("P_1",),
        "fc2.bias": ("P_res",),
        "fc3.bias": ("P_2",),
        "fc4.bias": ("P_res",),
        "fc5.bias": (None,),
        
    })

    # print(params_a)
    # print(params_b)

    # # find the permutation
    # perm = weight_matching(ps, params_a, params_b, silent=True)

    # # permute the params_b
    # updated_params = apply_permutation(ps, perm, params_b)

    perm = ot_weight_fusion(ps, params_a, params_b, exact=True, normalization=False, silent=True)
    updated_params = apply_permutation_by_matrix(ps, perm, params_b)

    model_b.load_state_dict(updated_params)

    print(torch.all(torch.isclose(model_a(x), model_b(x))))

    is_fail = 0
    for k, v in params_a.items():
        try:
            assert torch.all(torch.isclose(params_a[k], updated_params[k]))

        except:
            # print(k)
            # print(params_a[k])
            # print(updated_params[k])

            is_fail = 1

    correct += (is_fail == 0)

print(f"{correct}/{total} (correct/total)")