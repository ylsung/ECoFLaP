import torch
import torch.nn as nn
from copy import deepcopy

from lavis.compression.weight_matching import permutation_spec_from_axes_to_perm, weight_matching, apply_permutation, ot_weight_fusion, apply_permutation_by_matrix 
from lavis.compression.structural_pruning import structural_pruning


total = 1 # number of trials
correct = 0 # number of successful matching

for iteration in range(total):
    di = 3
    d1 = 5
    d2 = 6
    do = 2

    x = torch.randn(5, di)

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
        def __init__(self, d1, d2):
            super().__init__()
            self.fc0 = nn.Linear(di, d1)
            self.fc1 = nn.Linear(d1, d2)
            self.fc2 = nn.Linear(d2, d1)
            self.fc3 = nn.Linear(d1, d2)
            self.fc4 = nn.Linear(d2, d1)
            self.fc5 = nn.Linear(d1, do)
        
        def forward(self, x):
            out_0 = self.fc0(x)
            out_2 = self.fc2(self.fc1(out_0)) + out_0
            out_4 = self.fc4(self.fc3(out_2)) + out_2

            return self.fc5(out_4)

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

    model_a = Model(d1, d2)
    model_b = Model(d1, d2 // 2)

    old_diff = ((model_a(x) - model_b(x)) ** 2).mean()

    params_a = model_a.state_dict()

    perm = structural_pruning(ps, params_a, {"P_res": 1.0, "P_1": 0.5, "P_2": 0.5}, max_iter=100, silent=True)

    updated_params = apply_permutation(ps, perm, params_a)

    model_b.load_state_dict(updated_params)

    new_diff = ((model_a(x) - model_b(x)) ** 2).mean()

    print(old_diff, new_diff)

    for j in range(5):

        params_b = model_b.state_dict()

        perm = ot_weight_fusion(ps, params_b, params_a, silent=True)
        updated_params = apply_permutation_by_matrix(ps, perm, params_a)

        model_b.load_state_dict(updated_params)


        new_diff = ((model_a(x) - model_b(x)) ** 2).mean()

        print(old_diff, new_diff)

        old_diff = new_diff