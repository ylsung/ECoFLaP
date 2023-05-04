import torch
import torch.nn as nn
import random

from copy import deepcopy


def unstrct_pruning(importance_measure, keys_to_prune, ignore_layers, ratio):
    
    ignore_layers_dict = {l: 1 for l in ignore_layers}
    masks = {}

    for k, v in importance_measure.items():
        if k in keys_to_prune and k not in ignore_layers:
            top_k = int(v.numel() * ratio)

            _, top_indices = v.float().reshape(-1).topk(top_k, dim=-1)

            mask = torch.zeros((v.numel(),), dtype=bool, device=v.device) # 1D
            mask.scatter_(-1, top_indices, 1)

            mask = mask.reshape_as(v)

            masks[k] = mask
        else:
            masks[k] = torch.ones_like(v, dtype=bool, device=v.device)

    return masks


def t5_unstrct_pruning(transformer, distilled_transformer, importance_measure, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio, is_global=False, pruned_indices=None):

    ignore_layers = [
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
    ] # not used but may be used in the future

    for k in importance_measure.keys():
        if "layer_norm" in k:
            ignore_layers.append(k)

    if pruned_indices is not None:
        mask = pruned_indices
    else:
        if res_keep_ratio != 1:
            keys_to_prune = {k: 1 for k in importance_measure.keys()}
            ratio = res_keep_ratio
        elif attn_keep_ratio != 1:
            keys_to_prune = {k: 1 for k in importance_measure.keys() if "SelfAttention" in k or "EncDecAttention" in k}
            ratio = attn_keep_ratio
        elif ffn_keep_ratio:
            keys_to_prune = {k: 1 for k in importance_measure.keys() if "DenseReluDense" in k}
            ratio = ffn_keep_ratio

        mask = unstrct_pruning(importance_measure, keys_to_prune, ignore_layers, ratio)

        # if is_global:
        #     distilled_transformer = t5_modify_global_pruning(distilled_transformer, pruned_state_dict)

    pruned_state_dict = {}

    for k, v in transformer.state_dict().items():
        pruned_state_dict[k] = v * mask[k].type(v.dtype)

    distilled_transformer.load_state_dict(pruned_state_dict)

    return distilled_transformer, mask