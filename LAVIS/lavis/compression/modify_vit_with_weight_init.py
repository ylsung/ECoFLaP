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

from lavis.compression.weight_matching import permutation_spec_from_axes_to_perm, get_permuted_param, apply_permutation, ot_weight_fusion, apply_permutation_by_matrix
from lavis.compression.structural_pruning import structural_pruning
from lavis.models.blip2_models.blip2 import LayerNorm, disabled_train, Blip2Base

from lavis.models.eva_vit import convert_weights_to_fp16


def get_vit_ps(num_layers, num_heads):
    ps_dict = {
        "cls_token": (None, None, "P_vit_res"),
        "pos_embed": (None, None, "P_vit_res"),
        "patch_embed.proj.weight": ("P_vit_res", None, None),
        "patch_embed.proj.bias": ("P_vit_res",),
        **{f"blocks.{j}.norm1.weight": ("P_vit_res",) for j in range(num_layers)},
        **{f"blocks.{j}.norm1.bias": ("P_vit_res",) for j in range(num_layers)},
        # **{f"blocks.{j}.attn.q.weight.{i}": (f"P_vit_self_qk_{j}_{i}", "P_vit_res") for i in range(num_heads) for j in range(num_layers)},
        # **{f"blocks.{j}.attn.k.weight.{i}": (f"P_vit_self_qk_{j}_{i}", "P_vit_res") for i in range(num_heads) for j in range(num_layers)},
        # **{f"blocks.{j}.attn.q_bias.{i}": (f"P_vit_self_qk_{j}_{i}",) for i in range(num_heads) for j in range(num_layers)},
        # **{f"blocks.{j}.attn.v.weight.{i}": (f"P_vit_self_vo_{j}_{i}", "P_vit_res") for i in range(num_heads) for j in range(num_layers)},
        # **{f"blocks.{j}.attn.v_bias.{i}": (f"P_vit_self_vo_{j}_{i}",) for i in range(num_heads) for j in range(num_layers)},
        # **{f"blocks.{j}.attn.proj.weight.{i}": ("P_vit_res", f"P_vit_self_vo_{j}_{i}") for i in range(num_heads) for j in range(num_layers)},
        # **{f"blocks.{j}.attn.proj.bias.{i}": ("P_vit_res",) for i in range(num_heads) for j in range(num_layers)},
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


def vit_pruning(vit, distilled_vit, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio):
    state_dict = vit.state_dict()

    num_layers = vit.depth
    num_heads = vit.num_heads

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

    ignore_layers = []

    perm = structural_pruning(ps, state_dict, keep_ratio_dict, max_iter=100, silent=True)

    ignore_layers_weights = {}

    for l in ignore_layers:
        ignore_layers_weights[l] = state_dict.pop(l)
    
    pruned_state_dict = apply_permutation(ps, perm, state_dict)

    pruned_state_dict.update(ignore_layers_weights)

    distilled_vit.load_state_dict(pruned_state_dict)

    return distilled_vit


def vit_fusion(vit, distilled_vit, distill_merge_ratio=0.5, exact=True, normalization=False, metric="dot"):
    num_layers = vit.depth
    num_heads = vit.num_heads

    ps_dict = get_vit_ps(num_layers, num_heads)
    ps = permutation_spec_from_axes_to_perm(ps_dict)

    state_dict = vit.state_dict()
    distilled_state_dict = distilled_vit.state_dict()

    ignore_layers = []

    perm = ot_weight_fusion(
        ps, 
        distilled_state_dict, 
        state_dict, 
        max_iter=100, 
        exact=exact, 
        normalization=normalization, 
        metric=metric, 
        silent=True,
    )

    ignore_layers_weights = {}

    for l in ignore_layers:
        ignore_layers_weights[l] = state_dict.pop(l)
    
    fusion_state_dict = apply_permutation_by_matrix(ps, perm, state_dict)

    fusion_state_dict.update(ignore_layers_weights)
    
    distilled_state_dict = distilled_vit.state_dict()

    for k in distilled_state_dict.keys():
        distilled_state_dict[k] = (1 - distill_merge_ratio) * distilled_state_dict[k] + distill_merge_ratio * fusion_state_dict[k]

    distilled_vit.load_state_dict(distilled_state_dict)

    return distilled_vit



def vit_modify_with_weight_init(vit, petl_config, freeze_vit, precision, sampled_loader=None):
    device = list(vit.parameters())[0].device

    vit.to("cpu")
    if petl_config.vit_side_pretrained_weight is not None:
        num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = petl_config.vit_side_pretrained_weight.split("-")

        num_layers = int(num_layers)
        res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = float(res_keep_ratio), float(attn_keep_ratio), float(ffn_keep_ratio)

        distilled_vit = vit.__class__(
            img_size=vit.img_size,
            patch_size=vit.patch_size,
            use_mean_pooling=False,
            embed_dim=int(vit.embed_dim * res_keep_ratio),
            depth=num_layers,
            num_heads=vit.num_heads,
            mlp_ratio=vit.mlp_ratio * ffn_keep_ratio,
            qkv_bias=True,
            drop_path_rate=vit.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_checkpoint=vit.use_checkpoint,
        )

        print(vit.depth, distilled_vit.depth)

        # if petl_config.permute_before_merge:
        #     print("Start permutation (based on the first layer)...")
        #     vit = permute_based_on_first_layer(vit)

        # elif petl_config.permute_on_block_before_merge:
        #     print("Start permutation (based on block)...")
        #     vit = permute_based_on_block(vit, eval(petl_config.distilled_block_ids))

        weights = None

        if petl_config.distillation_init:
            # if "sum" in petl_config.distillation_init:

            #     weights = weights_interpolation_from_different_layers(
            #         vit.state_dict(), 
            #         distilled_transformer.state_dict().keys(),
            #         eval(petl_config.distilled_block_ids),
            #         eval(petl_config.distilled_block_weights) if petl_config.distilled_block_weights is not None else None,
            #         petl_config.modules_to_merge,
            #     )
            # elif "mul" in petl_config.distillation_init:
            #     weights = weights_multiplication_from_different_layers(
            #         vit.state_dict(), 
            #         distilled_transformer.state_dict().keys(),
            #         eval(petl_config.distilled_block_ids),
            #     )

            # elif "regmean-distill" in petl_config.distillation_init:
            #     cache_type = "input_output_gram"
            #     representations = get_activations(vit, sampled_loader, eval(petl_config.distilled_block_ids), cache_type=cache_type)
            #     weights = weights_regmean_distill_from_different_layers(
            #         vit.state_dict(), 
            #         distilled_transformer.state_dict().keys(),
            #         eval(petl_config.distilled_block_ids),
            #         representations,
            #         petl_config.scaling_factor
            #     )

            # elif "regmean" in petl_config.distillation_init:
            #     cache_type = "input_gram"
            #     representations = get_activations(vit, sampled_loader, eval(petl_config.distilled_block_ids), cache_type=cache_type)
            #     weights = weights_regmean_from_different_layers(
            #         vit.state_dict(), 
            #         distilled_transformer.state_dict().keys(),
            #         eval(petl_config.distilled_block_ids),
            #         representations,
            #         petl_config.scaling_factor
            #     )

            # if weights is not None:
            #     distilled_transformer.load_state_dict(weights)

            # if "rep" in petl_config.distillation_init:

            #     layer_ids_list = get_layer_ids(eval(petl_config.distilled_block_ids))
            #     print("rep_stack_forward:", layer_ids_list)
            #     distilled_transformer.encoder.forward = functools.partial(rep_stack_forward, distilled_transformer.encoder)
            #     distilled_transformer.decoder.forward = functools.partial(rep_stack_forward, distilled_transformer.decoder)

            #     distilled_transformer.encoder.layer_ids_list = layer_ids_list
            #     distilled_transformer.decoder.layer_ids_list = layer_ids_list

            # if "learnable" in petl_config.distillation_init:
            #     distilled_transformer = t5_modify_for_learnable_merge(
            #         vit, 
            #         distilled_transformer, 
            #         eval(petl_config.distilled_block_ids), 
            #         petl_config.learnable_weight_type
            #     )

            if "mag_prune" in petl_config.distillation_init:
                print("Apply magnitude pruning on ViT...")
                distilled_vit = vit_pruning(
                    vit, 
                    distilled_vit, 
                    res_keep_ratio, 
                    attn_keep_ratio, 
                    ffn_keep_ratio
                )

            if "fusion" in petl_config.distillation_init:
                print("Apply fusion on ViT...")
                distilled_vit = vit_fusion(
                    vit, 
                    distilled_vit, 
                    distill_merge_ratio=petl_config.distill_merge_ratio, 
                    exact=petl_config.exact, 
                    normalization=petl_config.normalization, 
                    metric=petl_config.metric
                )

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

        del vit
        return distilled_vit
    
    vit.to(device)
    return vit


if __name__ == "__main__":
    import torch
    import torch.nn as nn


    class Config:
        vit_side_pretrained_weight = "39-1.0-1.0-0.5"
        distillation_init = "mag_prune"
        distill_merge_ratio = 0.5
        exact = True
        normalization = False
        metric = "dot"

    config = Config()

    x = torch.randn(2, 3, 224, 224)
    vit, _ = Blip2Base.init_vision_encoder(
        "eva_clip_g", 224, 0, False, "fp32"
    )

    vit.eval()

    old_output = vit(x)

    vit = vit_modify_with_weight_init(vit, config, True, "fp32")

    new_output = vit(x)

    print("diff: ", torch.mean((old_output - new_output) ** 2))
        