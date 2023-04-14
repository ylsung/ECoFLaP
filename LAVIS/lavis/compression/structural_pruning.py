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

                # print(wk)
                # print(w_a.shape)

                # print(wk)
                # print("old")
                # if len(w_a.shape) == 1:
                #     w_a = w_a.unsqueeze(-1)
                #     A += torch.norm(w_a, dim=-1)
                #     print(torch.norm(w_a, dim=-1) ** 2)
                # else:
                #     A += torch.norm(w_a.T, dim=axis)

                #     print(w_a.shape)
                #     print(torch.norm(w_a.T, dim=axis) ** 2)

                w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1)).float()

                # print("new")
                # print(torch.diagonal(w_a @ w_a.T))

                A += torch.diagonal(w_a @ w_a.T)

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


def get_ps(config):
    encoder_layers = config.num_layers
    decoder_layers = config.num_decoder_layers
    num_heads = config.num_heads

    ps_dict = {
        "shared.weight": (None, "P_res"), # embedding layer, need to shuffle the index
        "encoder.embed_tokens.weight": (None, "P_res"),
        "encoder.final_layer_norm.weight": ("P_res",),
        "decoder.embed_tokens.weight": (None, "P_res"),
        "decoder.final_layer_norm.weight": ("P_res",),
        "lm_head.weight": (None, "P_res"),

        **{f"encoder.block.{j}.layer.0.SelfAttention.q.weight.{i}": (f"P_enc_self_qk_{j}_{i}", "P_res") for i in range(num_heads) for j in range(encoder_layers)},                                                                                                 
        **{f"encoder.block.{j}.layer.0.SelfAttention.k.weight.{i}": (f"P_enc_self_qk_{j}_{i}", "P_res") for i in range(num_heads) for j in range(encoder_layers)},                                                                                                 
        **{f"encoder.block.{j}.layer.0.SelfAttention.v.weight.{i}": (f"P_enc_self_vo_{j}_{i}", "P_res") for i in range(num_heads) for j in range(encoder_layers)},                                                                                                 
        **{f"encoder.block.{j}.layer.0.SelfAttention.o.weight.{i}": ("P_res", f"P_enc_self_vo_{j}_{i}") for i in range(num_heads) for j in range(encoder_layers)},                                                                                                 
        **{f"encoder.block.{j}.layer.0.layer_norm.weight": ("P_res",) for j in range(encoder_layers)},                                                                                                      
        **{f"encoder.block.{j}.layer.1.DenseReluDense.wo.weight": ("P_res", f"P_enc_ffn_{j}") for j in range(encoder_layers)},                                                                                               
        **{f"encoder.block.{j}.layer.1.layer_norm.weight": ("P_res",) for j in range(encoder_layers)},

        **{f"decoder.block.{j}.layer.0.SelfAttention.q.weight.{i}": (f"P_dec_self_qk_{j}_{i}", "P_res") for i in range(num_heads) for j in range(decoder_layers)},                                                                                                 
        **{f"decoder.block.{j}.layer.0.SelfAttention.k.weight.{i}": (f"P_dec_self_qk_{j}_{i}", "P_res") for i in range(num_heads) for j in range(decoder_layers)},                                                                                                 
        **{f"decoder.block.{j}.layer.0.SelfAttention.v.weight.{i}": (f"P_dec_self_vo_{j}_{i}", "P_res") for i in range(num_heads) for j in range(decoder_layers)},                                                                                                 
        **{f"decoder.block.{j}.layer.0.SelfAttention.o.weight.{i}": ("P_res", f"P_dec_self_vo_{j}_{i}") for i in range(num_heads) for j in range(decoder_layers)},                                                                                                 
        **{f"decoder.block.{j}.layer.0.layer_norm.weight": ("P_res",) for j in range(decoder_layers)},                                                                                                      
        **{f"decoder.block.{j}.layer.1.EncDecAttention.q.weight.{i}": (f"P_cross_qk_{j}_{i}", "P_res") for i in range(num_heads) for j in range(decoder_layers)},                                                                                               
        **{f"decoder.block.{j}.layer.1.EncDecAttention.k.weight.{i}": (f"P_cross_qk_{j}_{i}", "P_res") for i in range(num_heads) for j in range(decoder_layers)},                                                                                               
        **{f"decoder.block.{j}.layer.1.EncDecAttention.v.weight.{i}": (f"P_cross_vo_{j}_{i}", "P_res") for i in range(num_heads) for j in range(decoder_layers)},                                                                                               
        **{f"decoder.block.{j}.layer.1.EncDecAttention.o.weight.{i}": ("P_res", f"P_cross_vo_{j}_{i}") for i in range(num_heads) for j in range(decoder_layers)},                                                                                               
        **{f"decoder.block.{j}.layer.1.layer_norm.weight": ("P_res",) for j in range(decoder_layers)},                                                                                                      
        **{f"decoder.block.{j}.layer.2.DenseReluDense.wo.weight": ("P_res", f"P_dec_ffn_{j}") for j in range(decoder_layers)},                                                                                               
        **{f"decoder.block.{j}.layer.2.layer_norm.weight": ("P_res",) for j in range(decoder_layers)},
    }

    if "flan-t5" in config._name_or_path:
        for j in range(encoder_layers):
            ps_dict[f"encoder.block.{j}.layer.1.DenseReluDense.wi_0.weight"] = (f"P_enc_ffn_{j}", "P_res")
            ps_dict[f"encoder.block.{j}.layer.1.DenseReluDense.wi_1.weight"] = (f"P_enc_ffn_{j}", "P_res")

        for j in range(decoder_layers):
            ps_dict[f"decoder.block.{j}.layer.2.DenseReluDense.wi_0.weight"] = (f"P_dec_ffn_{j}", "P_res")
            ps_dict[f"decoder.block.{j}.layer.2.DenseReluDense.wi_1.weight"] = (f"P_dec_ffn_{j}", "P_res")

    elif "t5" in config._name_or_path:
        for j in range(encoder_layers):
            ps_dict[f"encoder.block.{j}.layer.1.DenseReluDense.wi.weight"] = (f"P_enc_ffn_{j}", "P_res")

        for j in range(decoder_layers):
            ps_dict[f"decoder.block.{j}.layer.2.DenseReluDense.wi.weight"] = (f"P_dec_ffn_{j}", "P_res")

    # print(ps_dict)
    ps = permutation_spec_from_axes_to_perm(ps_dict)

    return ps


def pruning(transformer, distilled_transformer, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio):
    state_dict = transformer.state_dict()

    encoder_layers = transformer.config.num_layers
    decoder_layers = transformer.config.num_decoder_layers
    num_heads = transformer.config.num_heads

    layers_with_heads = {
        "encoder": [
            "encoder.block.{}.layer.0.SelfAttention.q.weight",
            "encoder.block.{}.layer.0.SelfAttention.k.weight",
            "encoder.block.{}.layer.0.SelfAttention.v.weight",
            "encoder.block.{}.layer.0.SelfAttention.o.weight",
        ],
        "decoder": [
            "decoder.block.{}.layer.0.SelfAttention.q.weight",
            "decoder.block.{}.layer.0.SelfAttention.k.weight",
            "decoder.block.{}.layer.0.SelfAttention.v.weight",
            "decoder.block.{}.layer.0.SelfAttention.o.weight",
            "decoder.block.{}.layer.1.EncDecAttention.q.weight",
            "decoder.block.{}.layer.1.EncDecAttention.k.weight",
            "decoder.block.{}.layer.1.EncDecAttention.v.weight",
            "decoder.block.{}.layer.1.EncDecAttention.o.weight",
        ]
    }

    ps = get_ps(transformer.config)

    keep_ratio_dict = {}
    for p, axes in ps.perm_to_axes.items():
        if "res" in p:
            keep_ratio_dict[p] = res_keep_ratio
        elif "self" in p or "cross" in p:
            keep_ratio_dict[p] = attn_keep_ratio
        elif "ffn" in p:
            keep_ratio_dict[p] = ffn_keep_ratio
        else:
            raise ValueError("The pruned module is unknown.")

    # split weights for num_heads

    state_dict_with_split_heads = {}

    ignore_layers = [
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    ]

    for k, v in state_dict.items():
        if "Attention" in k and k not in ignore_layers:
            if k.endswith("o.weight"):
                weight_chunks = torch.chunk(v, num_heads, dim=1)
            else:
                weight_chunks = torch.chunk(v, num_heads, dim=0)

            for chunk_id in range(len(weight_chunks)):
                chunk_k = k + f".{chunk_id}"
                state_dict_with_split_heads[chunk_k] = weight_chunks[chunk_id]
        else:
            state_dict_with_split_heads[k] = v

    state_dict = state_dict_with_split_heads

    perm = structural_pruning(ps, state_dict, keep_ratio_dict, max_iter=100, silent=True)

    ignore_layers_weights = {}

    for l in ignore_layers:
        ignore_layers_weights[l] = state_dict.pop(l)
    
    pruned_state_dict = apply_permutation(ps, perm, state_dict)

    pruned_state_dict.update(ignore_layers_weights)

    # combine the weights of attention

    for module_type in ["encoder", "decoder"]:
        if module_type == "encoder":
            num_layers = encoder_layers
        else:
            num_layers = decoder_layers

        for i in range(num_layers):
            for layer_with_heads in layers_with_heads[module_type]:
                weights = []
                new_layer_with_heads = layer_with_heads.format(i)
                for head_idx in range(num_heads):
                    layer_with_heads_this_head = new_layer_with_heads + f".{head_idx}"
                    weights.append(pruned_state_dict[layer_with_heads_this_head])

                if new_layer_with_heads.endswith("o.weight"):
                    pruned_state_dict[new_layer_with_heads] = torch.cat(weights, dim=1)
                else:
                    pruned_state_dict[new_layer_with_heads] = torch.cat(weights, dim=0)

                for head_idx in range(num_heads):
                    layer_with_heads_this_head = new_layer_with_heads + f".{head_idx}"

                    del pruned_state_dict[layer_with_heads_this_head]

    
    distilled_transformer.load_state_dict(pruned_state_dict)

    return distilled_transformer


def fusion(transformer, distilled_transformer, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio):
    state_dict = transformer.state_dict()

    encoder_layers = transformer.config.num_layers
    decoder_layers = transformer.config.num_decoder_layers
    num_heads = transformer.config.num_heads

    layers_with_heads = {
        "encoder": [
            "encoder.block.{}.layer.0.SelfAttention.q.weight",
            "encoder.block.{}.layer.0.SelfAttention.k.weight",
            "encoder.block.{}.layer.0.SelfAttention.v.weight",
            "encoder.block.{}.layer.0.SelfAttention.o.weight",
        ],
        "decoder": [
            "decoder.block.{}.layer.0.SelfAttention.q.weight",
            "decoder.block.{}.layer.0.SelfAttention.k.weight",
            "decoder.block.{}.layer.0.SelfAttention.v.weight",
            "decoder.block.{}.layer.0.SelfAttention.o.weight",
            "decoder.block.{}.layer.1.EncDecAttention.q.weight",
            "decoder.block.{}.layer.1.EncDecAttention.k.weight",
            "decoder.block.{}.layer.1.EncDecAttention.v.weight",
            "decoder.block.{}.layer.1.EncDecAttention.o.weight",
        ]
    }

    ps = get_ps(transformer.config)

    keep_ratio_dict = {}
    for p, axes in ps.perm_to_axes.items():
        if "res" in p:
            keep_ratio_dict[p] = res_keep_ratio
        elif "self" in p or "cross" in p:
            keep_ratio_dict[p] = attn_keep_ratio
        elif "ffn" in p:
            keep_ratio_dict[p] = ffn_keep_ratio
        else:
            raise ValueError("The pruned module is unknown.")

    # split weights for num_heads

    state_dict_with_split_heads = {}

    ignore_layers = [
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    ]

    for k, v in state_dict.items():
        if "Attention" in k and k not in ignore_layers:
            if k.endswith("o.weight"):
                weight_chunks = torch.chunk(v, num_heads, dim=1)
            else:
                weight_chunks = torch.chunk(v, num_heads, dim=0)

            for chunk_id in range(len(weight_chunks)):
                chunk_k = k + f".{chunk_id}"
                state_dict_with_split_heads[chunk_k] = weight_chunks[chunk_id]
        else:
            state_dict_with_split_heads[k] = v

    state_dict = state_dict_with_split_heads

    perm = structural_pruning(ps, state_dict, keep_ratio_dict, max_iter=100, silent=False)

    ignore_layers_weights = {}

    for l in ignore_layers:
        ignore_layers_weights[l] = state_dict.pop(l)
    
    pruned_state_dict = apply_permutation(ps, perm, state_dict)

    pruned_state_dict.update(ignore_layers_weights)

    # combine the weights of attention

    for module_type in ["encoder", "decoder"]:
        if module_type == "encoder":
            num_layers = encoder_layers
        else:
            num_layers = decoder_layers

        for i in range(num_layers):
            for layer_with_heads in layers_with_heads[module_type]:
                weights = []
                new_layer_with_heads = layer_with_heads.format(i)
                for head_idx in range(num_heads):
                    layer_with_heads_this_head = new_layer_with_heads + f".{head_idx}"
                    weights.append(pruned_state_dict[layer_with_heads_this_head])

                if new_layer_with_heads.endswith("o.weight"):
                    pruned_state_dict[new_layer_with_heads] = torch.cat(weights, dim=1)
                else:
                    pruned_state_dict[new_layer_with_heads] = torch.cat(weights, dim=0)

                for head_idx in range(num_heads):
                    layer_with_heads_this_head = new_layer_with_heads + f".{head_idx}"

                    del pruned_state_dict[layer_with_heads_this_head]

    
    distilled_transformer.load_state_dict(pruned_state_dict)

    return distilled_transformer


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
         "l4.weight": (None, "P_res"),
         "l4.bias": (None,),
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

    param = model.state_dict()

    perm = structural_pruning(ps, param, 0.5, max_iter=100, silent=False)


    def total_norm(weights):
        total = 0

        for n, v in weights.items():
            total += (v ** 2).sum().item()

        return total

    pruned_weights = apply_permutation(ps, perm, param)

    print(perm)

    print(pruned_weights)

    selected_norm = total_norm(pruned_weights)

    print(selected_norm)

    print("Random")

    biggest_random = 0
    smallest_random = 1000
    for i in range(10000):
        random_perm = {'P_res': torch.randperm(10)[:5], 'P_l2': torch.randperm(6)[:3]}

        random_pruned_weights = apply_permutation(ps, random_perm, param)

        random_norm = total_norm(random_pruned_weights)

        biggest_random = max(biggest_random, random_norm)

        smallest_random = min(smallest_random, random_norm)

    print(biggest_random, smallest_random)
        