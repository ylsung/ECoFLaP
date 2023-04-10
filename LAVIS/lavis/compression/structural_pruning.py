import re
import random
import torch
import torch.nn as nn
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

from typing import NamedTuple

from lavis.compression.weight_matching import permutation_spec_from_axes_to_perm, get_permuted_param, apply_permutation


def get_pruned_dim(dim, keep_ratio):
    pruned_dim = int(dim * keep_ratio)

    return pruned_dim


def structural_pruning(ps, params_a, keep_ratio, max_iter=100, silent=True):
    """Find a permutation of `params_b` to make them match `params_a`."""
    orig_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    perm_sizes = {p: get_pruned_dim(params_a[axes[0][0]].shape[axes[0][1]], keep_ratio) for p, axes in ps.perm_to_axes.items()}

    if not silent:
        print(perm_sizes)

    perm = {p: np.random.choice(orig_sizes[p], n, replace=False) for p, n in perm_sizes.items()}
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            n = orig_sizes[p]
            selected_n = perm_sizes[p]
            A = torch.zeros((n, ))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = get_permuted_param(ps, perm, wk, params_a, except_axis=axis)

                A += np.linalg.norm(w_a, axis=axis)

            ci = np.argmax(A, -1)

            selected_ci = ci[:selected_n]
            old_selected_ci = perm[p]

            oldL = A[old_selected_ci.long()].sum()
            newL = A[selected_ci.long()].sum()

            if not silent:
                print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = torch.Tensor(selected_ci)

        if not progress:
            break

    # perm = {name: p[name][:perm_sizes[name]] for name, p in perm.items()}

    return perm


if __name__ == "__main__":
    import torch
    import torch.nn as nn


    ps = permutation_spec_from_axes_to_perm(
        {"l1.weight": ("P_res", None),
         "l1.bias": ("P_res",),
         "l2.weight": ("P_l2", "P_res"),
         "l2.bias": ("P_l2",),
         "l3.weight": ("P_res", "P_l2"),
         "l3.bias": ("P_res",),
         "l4.weight": ("P_l4", "P_res"),
         "l4.bias": ("P_l4",),
        }
    )


    class Model(nn.Module):
        def __init__(self):

            super().__init__()

            self.l1 = nn.Linear(6, 10)
            self.l2 = nn.Linear(10, 6)
            self.l3 = nn.Linear(6, 10)
            self.l4 = nn.Linear(10, 6)

        def forward(self, x):
            return self.l3(x + self.l2(self.l1(x)))


    model = Model()

    param = list(model.named_parameters())

    print(param)

    perm = structural_pruning(ps, param, 0.5, max_iter=100, silent=False)

    print(perm)
    