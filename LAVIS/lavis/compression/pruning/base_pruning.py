import re
import random
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

from typing import NamedTuple

from lavis.compression.weight_matching import permutation_spec_from_axes_to_perm, get_permuted_param, apply_permutation, ot_weight_fusion, get_permuted_param_by_matrix, apply_permutation_by_matrix
from lavis.compression.global_pruning_model_modifier.t5_model_modifier import t5_modify_global_pruning


def get_pruned_dim(dim, keep_ratio):
    pruned_dim = int(dim * keep_ratio)

    return pruned_dim


def structural_pruning(ps, params_a, keep_ratio_dict, max_iter=100, silent=True):
    """Find a permutation of `params_b` to make them match `params_a`."""
    orig_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    perm_sizes = {p: get_pruned_dim(params_a[axes[0][0]].shape[axes[0][1]], keep_ratio_dict[p]) for p, axes in ps.perm_to_axes.items()}

    if not silent:
        print(perm_sizes)

    perm = {p: torch.sort(torch.randperm(orig_sizes[p])[:n])[0] for p, n in perm_sizes.items()}
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            n = orig_sizes[p]
            selected_n = perm_sizes[p]
            A = torch.zeros((n, ))
            for wk, axis in ps.perm_to_axes[p]:

                # print(n, p, wk, axis)

                # print(perm[p].shape)
    
                w_a = get_permuted_param(ps, perm, wk, params_a, except_axis=axis)

                w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1)).float()

                A += torch.norm(w_a, p=1, dim=-1)

            # print(A)

            ci = torch.argsort(A, -1, descending=True)

            # print(ci)

            selected_ci = ci[:selected_n]

            selected_ci = torch.sort(selected_ci)[0]

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


def get_top_k_globally_grouping(D, topk, groups):
    # Select at least one value from each tensor in D
    
    max_length = 0
    for k in D.keys():
        max_length += len(D[k])
    assert len(D) <= topk <= max_length, "The length should be within range"

    keys = list(D.keys())
    # Get the indices of the top k-m values in the concatenated tensor
    top_indices = torch.topk(torch.cat([D[k] for k in keys]), topk).indices

    
    selected_indices = {k:[] for k in D.keys()}

    accum_length = []
    for key, d in D.items():
        if len(accum_length) == 0:
            accum_length.append(len(d))
        else:
            accum_length.append(len(d) + accum_length[-1])

    # Split the selected values tensor back into the original tensors in D
    
    
    for top_index in top_indices:
        pre_len = 0
        for key, acc_len in zip(keys, accum_length):
            if top_index < acc_len:
                selected_indices[key].append(top_index.unsqueeze(-1) - pre_len)
                break
            else:
                pre_len = acc_len

    # compute the number of selection for each group

    num_selected_group = {}

    for group in groups:
        n_selection = 0
        for g in group:
            if g not in selected_indices:
                continue
            n_selection += len(selected_indices[g])

        n_selection = round(n_selection / len(group))
        n_selection = max(1, n_selection)

        for g in group:
            if g not in selected_indices:
                continue
            num_selected_group[g] = n_selection

    selected_values = []

    # reset it
    selected_indices = {k:[] for k in D.keys()}

    all_values_without_top_in_tensor = []

    num_selected = 0
    for key, d in D.items():
        if key not in num_selected_group:
            top_indices = torch.topk(d, 1).indices
            # fill the selected with larger values, making all_values_without_top_in_tensor be less than zero for those values
            selected_values = torch.zeros_like(d).scatter_(0, top_indices, d[top_indices] + 10)

            num_selected += 1
        else:
            top_indices = torch.topk(d, num_selected_group[key]).indices
            selected_values = d + 10

            num_selected += num_selected_group[key]

        selected_indices[key].append(top_indices)

        all_values_without_top_in_tensor.append(d - selected_values)

    # Concatenate the selected values into a single tensor
    all_values_without_top_in_tensor = torch.cat(all_values_without_top_in_tensor)

    rest_num_selected = max(0, topk - num_selected)
    # Get the indices of the top k-m values in the concatenated tensor
    top_indices = torch.topk(all_values_without_top_in_tensor, rest_num_selected).indices

    # Split the selected values tensor back into the original tensors in D
    keys = list(D.keys())
    
    for top_index in top_indices:
        pre_len = 0
        for key, acc_len in zip(keys, accum_length):
            if top_index < acc_len:
                selected_indices[key].append(top_index.unsqueeze(-1) - pre_len)
                break
            else:
                pre_len = acc_len

    selected_indices = {k: torch.sort(torch.cat(v)).values for k, v in selected_indices.items()}
    
    return selected_indices


def get_top_k_globally(D, topk):
    # Select at least one value from each tensor in D
    
    max_length = 0
    for k in D.keys():
        max_length += len(D[k])
    assert len(D) <= topk <= max_length, "The length should be within range"
    selected_values = []

    selected_indices = {k:[] for k in D.keys()}

    all_values_without_top_in_tensor = []

    accum_length = []
    for key, d in D.items():
        top_indices = torch.topk(d, 1).indices

        selected_indices[key].append(top_indices)

        # fill the selected with larger values, making all_values_without_top_in_tensor be less than zero for those values
        selected_values = torch.zeros_like(d).scatter_(0, top_indices, d[top_indices] + 10)

        all_values_without_top_in_tensor.append(d - selected_values)

        if len(accum_length) == 0:
            accum_length.append(len(d))
        else:
            accum_length.append(len(d) + accum_length[-1])
    
    # Concatenate the selected values into a single tensor
    all_values_without_top_in_tensor = torch.cat(all_values_without_top_in_tensor)

    # Get the indices of the top k-m values in the concatenated tensor
    top_indices = torch.topk(all_values_without_top_in_tensor, topk - len(D)).indices

    # Split the selected values tensor back into the original tensors in D
    keys = list(D.keys())
    
    for top_index in top_indices:
        pre_len = 0
        for key, acc_len in zip(keys, accum_length):
            if top_index < acc_len:
                selected_indices[key].append(top_index.unsqueeze(-1) - pre_len)
                break
            else:
                pre_len = acc_len

    selected_indices = {k: torch.sort(torch.cat(v)).values for k, v in selected_indices.items()}
    
    return selected_indices

def compute_L_globally(A, selected_dict):
    L = 0

    for k in A.keys():
        s_idx = selected_dict[k]
        L += A[k][s_idx.long()].sum()

    return L


def global_structural_pruning(ps, params_a, keep_ratio_dict, groups=None, max_iter=100, silent=True):
    """Find a permutation of `params_b` to make them match `params_a`."""
    orig_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    perm_sizes = {p: get_pruned_dim(params_a[axes[0][0]].shape[axes[0][1]], keep_ratio_dict[p]) for p, axes in ps.perm_to_axes.items()}
    p_is_prune = {k: orig_sizes[k] != perm_sizes[k] for k in perm_sizes.keys()}

    total_dim_to_prune = 0

    for k in perm_sizes.keys():
        total_dim_to_prune += abs(orig_sizes[k] - perm_sizes[k])

    if not silent:
        print(perm_sizes)

    perm = {p: torch.sort(torch.randperm(orig_sizes[p])[:n])[0] for p, n in perm_sizes.items()}
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False

        A_dict = {}
        # get all As
        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]

            if not p_is_prune[p]:
                continue

            n = orig_sizes[p]
            selected_n = perm_sizes[p]
            A = torch.zeros((n, ))
            for wk, axis in ps.perm_to_axes[p]:

                # print(n, p, wk, axis)

                # print(perm[p].shape)
    
                w_a = get_permuted_param(ps, perm, wk, params_a, except_axis=axis)

                w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1)).float()

                A += torch.norm(w_a, p=1, dim=-1)

            # print(A)

            A_dict[p] = A

        if groups is not None:
            selected_dict = get_top_k_globally_grouping(A_dict, total_dim_to_prune, groups)
        else:
            selected_dict = get_top_k_globally(A_dict, total_dim_to_prune)

        filtered_old_selected_dict = {p: v for p, v in perm.items() if p in selected_dict}

        newL = compute_L_globally(A_dict, selected_dict)
        oldL = compute_L_globally(A_dict, filtered_old_selected_dict)

        if not silent:
            print(f"{iteration}/{p}: {newL - oldL}")
        progress = progress or newL > oldL + 1e-12 # greater is better

        perm.update(selected_dict)

        if not progress:
            break

    return perm


def unstructural_pruning(importance_measure, keys_to_prune, ignore_layers, ratio):
    ignore_layers_dict = {l: 1 for l in ignore_layers}
    masks = {}

    for k, v in importance_measure.items():
        if k in keys_to_prune and k not in ignore_layers_dict:
            top_k = int(v.numel() * ratio)

            _, top_indices = v.float().reshape(-1).topk(top_k, dim=-1)

            mask = torch.zeros((v.numel(),), dtype=bool, device=v.device) # 1D
            mask.scatter_(-1, top_indices, 1)

            mask = mask.reshape_as(v)

            masks[k] = mask
        else:
            masks[k] = torch.ones_like(v, dtype=bool, device=v.device)

    return masks