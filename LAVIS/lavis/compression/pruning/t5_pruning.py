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

from lavis.compression.pruning.base_pruning import global_structural_pruning, structural_pruning, unstructural_pruning


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

    groups = []

    for j in range(encoder_layers):
        groups.append([])
        for i in range(num_heads):
            groups[-1].append(f"P_enc_self_qk_{j}_{i}")

    for j in range(encoder_layers):
        groups.append([])
        for i in range(num_heads):
            groups[-1].append(f"P_enc_self_vo_{j}_{i}")

    for j in range(decoder_layers):
        groups.append([])
        for i in range(num_heads):
            groups[-1].append(f"P_dec_self_qk_{j}_{i}")

    for j in range(decoder_layers):
        groups.append([])
        for i in range(num_heads):
            groups[-1].append(f"P_dec_self_vo_{j}_{i}")

    for j in range(decoder_layers):
        groups.append([])
        for i in range(num_heads):
            groups[-1].append(f"P_cross_qk_{j}_{i}")

    for j in range(decoder_layers):
        groups.append([])
        for i in range(num_heads):
            groups[-1].append(f"P_cross_vo_{j}_{i}")

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

    return ps_dict, groups


def split_weights_for_heads(state_dict, ignore_layers, num_heads):
    state_dict_with_split_heads = {}
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

    return state_dict_with_split_heads


def t5_strct_pruning(transformer, importance_measure, keep_indices_or_masks, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio, ignore_layers=[], is_global=False):
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

    ps_dict, groups = get_ps(transformer.config)
    ps = permutation_spec_from_axes_to_perm(ps_dict)

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

    # ignore_layers = [
    #     "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
    #     "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    # ]

    state_dict = split_weights_for_heads(
        transformer.state_dict(), ignore_layers, num_heads
    )

    if keep_indices_or_masks is None:
        importance_measure = split_weights_for_heads(
            importance_measure, ignore_layers, num_heads
        )
        if is_global:
            perm = global_structural_pruning(ps, importance_measure, keep_ratio_dict, groups=groups, max_iter=100, silent=True)
        else:
            perm = structural_pruning(ps, importance_measure, keep_ratio_dict, max_iter=100, silent=True)
    else:
        # use cached indices/masks
        perm = keep_indices_or_masks

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

    return pruned_state_dict, perm


def t5_unstrct_pruning(transformer, importance_measure, keep_indices_or_masks, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio, ignore_layers=[], is_global=False):
    # ignore_layers = [
    #     "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
    #     "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
    # ] # not used but may be used in the future

    # for k in importance_measure.keys():
    #     # don't prune embedding layers and lm_head
    #     if any(sub_n in k for sub_n in ["shared", "embed_tokens", "lm_head", "layer_norm"]):
    #         ignore_layers.append(k)

    if keep_indices_or_masks is None:
        if res_keep_ratio != 1:
            keys_to_prune = {k: 1 for k in importance_measure.keys()}
            ratio = res_keep_ratio
        elif attn_keep_ratio != 1:
            keys_to_prune = {k: 1 for k in importance_measure.keys() if "SelfAttention" in k or "EncDecAttention" in k}
            ratio = attn_keep_ratio
        elif ffn_keep_ratio:
            keys_to_prune = {k: 1 for k in importance_measure.keys() if "DenseReluDense" in k}
            ratio = ffn_keep_ratio

        # for k, v in transformer.state_dict().items():
        #     print(v.device, importance_measure[k].device)
        mask = unstructural_pruning(importance_measure, keys_to_prune, ignore_layers, ratio)
    else:
        # use cached mask/indices
        mask = keep_indices_or_masks
    # if is_global:
    #     distilled_transformer = t5_modify_global_pruning(distilled_transformer, pruned_state_dict)

    pruned_state_dict = {}

    for k, v in transformer.state_dict().items():
        pruned_state_dict[k] = v * mask[k].type(v.dtype)

    return pruned_state_dict, mask


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