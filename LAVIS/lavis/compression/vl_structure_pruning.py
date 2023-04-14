import re
import random
import torch
import torch.nn as nn
from functools import partial
import numpy as np
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

from typing import NamedTuple

from lavis.compression.weight_matching import permutation_spec_from_axes_to_perm, get_permuted_param, apply_permutation
from lavis.models.blip2_models.blip2 import LayerNorm, disabled_train, Blip2Base

from lavis.models.eva_vit import convert_weights_to_fp16


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


def get_vit_ps(num_layers, num_heads):
    ps_dict = {
        "cls_token": (None, None, "P_vit_res"),
        "pos_embed": (None, None, "P_vit_res"),
        "patch_embed.proj.weight": ("P_vit_res", None, None),
        "patch_embed.proj.bias": ("P_vit_res",),
        **{f"blocks.{j}.norm1.weight": ("P_vit_res",) for j in range(num_layers)},
        **{f"blocks.{j}.norm1.bias": ("P_vit_res",) for j in range(num_layers)},
        # **{f"visual_encoder.blocks.{j}.attn.q.weight.{i}": (f"P_vit_self_qk_{j}_{i}", "P_vit_res") for i in range(num_heads) for j in range(num_layers)},
        # **{f"visual_encoder.blocks.{j}.attn.k.weight.{i}": (f"P_vit_self_qk_{j}_{i}", "P_vit_res") for i in range(num_heads) for j in range(num_layers)},
        # **{f"visual_encoder.blocks.{j}.attn.q_bias.{i}": (f"P_vit_self_qk_{j}_{i}",) for i in range(num_heads) for j in range(num_layers)},
        # **{f"visual_encoder.blocks.{j}.attn.v.weight.{i}": (f"P_vit_self_vo_{j}_{i}", "P_vit_res") for i in range(num_heads) for j in range(num_layers)},
        # **{f"visual_encoder.blocks.{j}.attn.v_bias.{i}": (f"P_vit_self_vo_{j}_{i}",) for i in range(num_heads) for j in range(num_layers)},
        # **{f"visual_encoder.blocks.{j}.attn.proj.weight.{i}": ("P_vit_res", f"P_vit_self_vo_{j}_{i}") for i in range(num_heads) for j in range(num_layers)},
        # **{f"visual_encoder.blocks.{j}.attn.proj.bias.{i}": ("P_vit_res",) for i in range(num_heads) for j in range(num_layers)},
        **{f"blocks.{j}.attn.qkv.weight": (None, "P_vit_res") for j in range(num_layers)},
        **{f"blocks.{j}.attn.q_bias": (None,) for j in range(num_layers)},
        **{f"blocks.{j}.attn.v_bias": (None,) for j in range(num_layers)},
        **{f"blocks.{j}.attn.proj.weight": ("P_vit_res", None) for j in range(num_layers)},
        **{f"blocks.{j}.attn.proj.bias": ("P_vit_res",) for j in range(num_layers)},

        **{f"blocks.{j}.norm2.weight": ("P_vit_res",) for j in range(num_layers)},
        **{f"blocks.{j}.norm2.bias": ("P_vit_res",) for j in range(num_layers)},
        **{f"blocks.{j}.mlp.fc1.weight": (f"P_vit_ffn_{j}", "P_vit_res") for j in range(num_layers)},
        **{f"blocks.{j}.mlp.fc1.bias": (f"P_vit_ffn_{j}",) for j in range(num_layers)},
        **{f"blocks.{j}.mlp.fc2.weight": ("P_vit_res", f"P_vit_ffn_{j}") for j in range(num_layers)},
        **{f"blocks.{j}.mlp.fc2.bias": ("P_vit_res",) for j in range(num_layers)},
    }

    # print(ps_dict)

    return ps_dict


def v_pruning(visual_encoder, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio, freeze_vit, precision):
    device = list(visual_encoder.parameters())[0].device
    visual_encoder.to("cpu")

    state_dict = visual_encoder.state_dict()

    num_layers = visual_encoder.depth
    num_heads = visual_encoder.num_heads

    distilled_vit = visual_encoder.__class__(
        img_size=visual_encoder.img_size,
        patch_size=visual_encoder.patch_size,
        use_mean_pooling=False,
        embed_dim=visual_encoder.embed_dim,
        depth=visual_encoder.depth,
        num_heads=visual_encoder.num_heads,
        mlp_ratio=visual_encoder.mlp_ratio * ffn_keep_ratio,
        qkv_bias=True,
        drop_path_rate=visual_encoder.drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=visual_encoder.use_checkpoint,
    )

    ps_dict = get_vit_ps(num_layers, num_heads)
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

    state_dict_with_split_heads = {}

    ignore_layers = [
        # "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        # "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    ]

    # for k, v in state_dict.items():
    #     if "Attention" in k and k not in ignore_layers:
    #         if k.endswith("o.weight"):
    #             weight_chunks = torch.chunk(v, num_heads, dim=1)
    #         else:
    #             weight_chunks = torch.chunk(v, num_heads, dim=0)

    #         for chunk_id in range(len(weight_chunks)):
    #             chunk_k = k + f".{chunk_id}"
    #             state_dict_with_split_heads[chunk_k] = weight_chunks[chunk_id]
    #     else:
    #         state_dict_with_split_heads[k] = v

    # state_dict = state_dict_with_split_heads

    perm = structural_pruning(ps, state_dict, keep_ratio_dict, max_iter=100, silent=True)

    ignore_layers_weights = {}

    for l in ignore_layers:
        ignore_layers_weights[l] = state_dict.pop(l)
    
    pruned_state_dict = apply_permutation(ps, perm, state_dict)

    pruned_state_dict.update(ignore_layers_weights)

    # # combine the weights of attention

    # for module_type in ["encoder", "decoder"]:
    #     if module_type == "encoder":
    #         num_layers = encoder_layers
    #     else:
    #         num_layers = decoder_layers

    #     for i in range(num_layers):
    #         for layer_with_heads in layers_with_heads[module_type]:
    #             weights = []
    #             new_layer_with_heads = layer_with_heads.format(i)
    #             for head_idx in range(num_heads):
    #                 layer_with_heads_this_head = new_layer_with_heads + f".{head_idx}"
    #                 weights.append(pruned_state_dict[layer_with_heads_this_head])

    #             if new_layer_with_heads.endswith("o.weight"):
    #                 pruned_state_dict[new_layer_with_heads] = torch.cat(weights, dim=1)
    #             else:
    #                 pruned_state_dict[new_layer_with_heads] = torch.cat(weights, dim=0)

    #             for head_idx in range(num_heads):
    #                 layer_with_heads_this_head = new_layer_with_heads + f".{head_idx}"

    #                 del pruned_state_dict[layer_with_heads_this_head]

    
    distilled_vit.load_state_dict(pruned_state_dict)

    distilled_vit.to(device)

    if precision == "fp16":
#         model.to("cuda") 
        convert_weights_to_fp16(distilled_vit)

    if freeze_vit:
        for name, param in distilled_vit.named_parameters():
            param.requires_grad = False
        distilled_vit = distilled_vit.eval()
        distilled_vit.train = disabled_train
        print("freeze distilled vision encoder")

    return distilled_vit


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    x = torch.randn(2, 3, 224, 224)
    visual_encoder, _ = Blip2Base.init_vision_encoder(
        "eva_clip_g", 224, 0, False, "fp32"
    )

    visual_encoder.eval()

    old_output = visual_encoder(x)

    visual_encoder = v_pruning(visual_encoder, 1.0, 1.0, 1.0, True, "fp32")

    new_output = visual_encoder(x)

    print("diff: ", torch.mean((old_output - new_output) ** 2))
        