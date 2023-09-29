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
from lavis.compression.pruning.base_pruning import structural_pruning, global_structural_pruning, unstructural_pruning
from lavis.compression.global_pruning_model_modifier.vit_model_modifier import vit_modify_global_pruning
from lavis.models.blip2_models.blip2 import LayerNorm, disabled_train, Blip2Base

from lavis.models.eva_vit import convert_weights_to_fp16


def get_vit_ps(num_layers, num_heads, has_norm, has_fc_norm, has_head):
    ps_dict = {
        "cls_token": (None, None, "P_vit_res"),
        "pos_embed": (None, None, "P_vit_res"),
        "patch_embed.proj.weight": ("P_vit_res", None, None),
        "patch_embed.proj.bias": ("P_vit_res",),
        **{f"blocks.{j}.norm1.weight": ("P_vit_res",) for j in range(num_layers)},
        **{f"blocks.{j}.norm1.bias": ("P_vit_res",) for j in range(num_layers)},

        **{f"blocks.{j}.attn.qkv.weight.q.{i}": (f"P_vit_self_qk_{j}_{i}", "P_vit_res") for i in range(num_heads) for j in range(num_layers)},
        **{f"blocks.{j}.attn.qkv.weight.k.{i}": (f"P_vit_self_qk_{j}_{i}", "P_vit_res") for i in range(num_heads) for j in range(num_layers)},
        **{f"blocks.{j}.attn.qkv.weight.v.{i}": (f"P_vit_self_vo_{j}_{i}", "P_vit_res") for i in range(num_heads) for j in range(num_layers)},
        **{f"blocks.{j}.attn.q_bias.{i}": (f"P_vit_self_qk_{j}_{i}",) for i in range(num_heads) for j in range(num_layers)},
        **{f"blocks.{j}.attn.v_bias.{i}": (f"P_vit_self_vo_{j}_{i}",) for i in range(num_heads) for j in range(num_layers)},
        **{f"blocks.{j}.attn.proj.weight.{i}": ("P_vit_res", f"P_vit_self_vo_{j}_{i}") for i in range(num_heads) for j in range(num_layers)},
        **{f"blocks.{j}.attn.proj.bias": ("P_vit_res",) for j in range(num_layers)},

        # **{f"blocks.{j}.attn.qkv.weight": (None, "P_vit_res") for j in range(num_layers)},
        # **{f"blocks.{j}.attn.q_bias": (None,) for j in range(num_layers)},
        # **{f"blocks.{j}.attn.v_bias": (None,) for j in range(num_layers)},
        # **{f"blocks.{j}.attn.proj.weight": ("P_vit_res", None) for j in range(num_layers)},
        # **{f"blocks.{j}.attn.proj.bias": ("P_vit_res",) for j in range(num_layers)},

        **{f"blocks.{j}.norm2.weight": ("P_vit_res",) for j in range(num_layers)},
        **{f"blocks.{j}.norm2.bias": ("P_vit_res",) for j in range(num_layers)},
        **{f"blocks.{j}.mlp.fc1.weight": (f"P_vit_ffn_{j}", "P_vit_res") for j in range(num_layers)},
        **{f"blocks.{j}.mlp.fc1.bias": (f"P_vit_ffn_{j}",) for j in range(num_layers)},
        **{f"blocks.{j}.mlp.fc2.weight": ("P_vit_res", f"P_vit_ffn_{j}") for j in range(num_layers)},
        **{f"blocks.{j}.mlp.fc2.bias": ("P_vit_res",) for j in range(num_layers)},
    }

    if has_norm:
        ps_dict["norm.weight"] = ("P_vit_res",)
        ps_dict["norm.bias"] = ("P_vit_res",)
    
    if has_fc_norm:
        ps_dict["fc_norm.weight"] = ("P_vit_res",)
        ps_dict["fc_norm.bias"] = ("P_vit_res",)

    if has_head:
        ps_dict["head.weight"] = (None, "P_vit_res")
        ps_dict["head.bias"] = (None,)

    # print(ps_dict)

    return ps_dict


def merge_weights_for_qkv(state_dict, num_layers):
    layers_with_heads = [
        "blocks.{}.attn.qkv.weight",
    ]

    name_list = ["q", "k", "v"]
    for i in range(num_layers):
        for layer_with_heads in layers_with_heads:
            weights = []
            new_layer_with_heads = layer_with_heads.format(i)
            for head_idx in name_list:
                layer_with_heads_this_head = new_layer_with_heads + f".{head_idx}"
                weights.append(state_dict[layer_with_heads_this_head])

            state_dict[new_layer_with_heads] = torch.cat(weights, dim=0)

            for head_idx in name_list:
                layer_with_heads_this_head = new_layer_with_heads + f".{head_idx}"

                del state_dict[layer_with_heads_this_head]

    return state_dict


def split_weights_for_qkv(state_dict):
    state_dict_with_split_qkv = {}
    name_list = ["q", "k", "v"]
    for k, v in state_dict.items():
        if "attn.qkv.weight" in k:
            weight_chunks = torch.chunk(v, 3, dim=0)

            for chunk_id in range(len(weight_chunks)):
                chunk_k = k + f".{name_list[chunk_id]}"
                state_dict_with_split_qkv[chunk_k] = weight_chunks[chunk_id]
        else:
            state_dict_with_split_qkv[k] = v

    return state_dict_with_split_qkv


def merge_weights_for_heads(state_dict, num_layers, num_heads):
    layers_with_heads = [
        "blocks.{}.attn.qkv.weight.q",
        "blocks.{}.attn.qkv.weight.k",
        "blocks.{}.attn.qkv.weight.v",
        "blocks.{}.attn.q_bias",
        "blocks.{}.attn.v_bias",
        "blocks.{}.attn.proj.weight",
    ]
    for i in range(num_layers):
        for layer_with_heads in layers_with_heads:
            weights = []
            new_layer_with_heads = layer_with_heads.format(i)
            for head_idx in range(num_heads):
                layer_with_heads_this_head = new_layer_with_heads + f".{head_idx}"
                weights.append(state_dict[layer_with_heads_this_head])

            if new_layer_with_heads.endswith("proj.weight"):
                state_dict[new_layer_with_heads] = torch.cat(weights, dim=1)
            else:
                state_dict[new_layer_with_heads] = torch.cat(weights, dim=0)

            for head_idx in range(num_heads):
                layer_with_heads_this_head = new_layer_with_heads + f".{head_idx}"

                del state_dict[layer_with_heads_this_head]

    return state_dict


def split_weights_for_heads(state_dict, num_heads):
    state_dict_with_split_heads = {}
    for k, v in state_dict.items():
        if "attn.qkv.weight" in k or "q_bias" in k or "v_bias" in k:
            weight_chunks = torch.chunk(v, num_heads, dim=0)

            for chunk_id in range(len(weight_chunks)):
                chunk_k = k + f".{chunk_id}"
                state_dict_with_split_heads[chunk_k] = weight_chunks[chunk_id]

        elif "attn.proj.weight" in k:
            weight_chunks = torch.chunk(v, num_heads, dim=1)

            for chunk_id in range(len(weight_chunks)):
                chunk_k = k + f".{chunk_id}"
                state_dict_with_split_heads[chunk_k] = weight_chunks[chunk_id]
        else:
            state_dict_with_split_heads[k] = v

    return state_dict_with_split_heads


def vit_strct_pruning(vit, importance_measure, keep_indices_or_masks, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio, ignore_layers=[], is_global=False):
    state_dict = vit.state_dict()

    num_layers = vit.depth
    num_heads = vit.num_heads

    # for clip models, blip's vit does not have the following three
    has_norm = getattr(vit, "norm", None)
    has_fc_norm = getattr(vit, "fc_norm", None)
    has_head = getattr(vit, "head", None)

    ps_dict = get_vit_ps(num_layers, num_heads, has_norm, has_fc_norm, has_head)
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
    # ignore_layers = []

    state_dict = split_weights_for_qkv(state_dict)
    state_dict = split_weights_for_heads(state_dict, num_heads)

    importance_measure = split_weights_for_qkv(importance_measure)
    
    if keep_indices_or_masks is None:
        importance_measure = split_weights_for_heads(importance_measure, num_heads)

        if is_global:
            perm = global_structural_pruning(ps, importance_measure, keep_ratio_dict, max_iter=100, silent=True)
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

    pruned_state_dict = merge_weights_for_heads(pruned_state_dict, num_layers, num_heads)
    pruned_state_dict = merge_weights_for_qkv(pruned_state_dict, num_layers)

    return pruned_state_dict, perm


def vit_fusion(vit, distilled_vit, distill_merge_ratio=0.5, exact=True, normalization=False, metric="dot", to_one=False, importance=False):
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
        to_one=to_one,
        importance=importance,
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


def vit_unstrct_pruning(vit, importance_measure, keep_indices_or_masks, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio, ignore_layers=[], is_global=False):

    # ignore_layers = []

    # for k in importance_measure.keys():
    #     # don't prune embedding layers and output layers
    #     if any(sub_n in k for sub_n in ["cls_token", "pos_embed", "patch_embed", "norm"]):
    #         ignore_layers.append(k)

    if keep_indices_or_masks is None:
        if res_keep_ratio != 1:
            keys_to_prune = {k: 1 for k in importance_measure.keys()}
            ratio = res_keep_ratio
        elif attn_keep_ratio != 1:
            keys_to_prune = {k: 1 for k in importance_measure.keys() if ".attn." in k}
            ratio = attn_keep_ratio
        elif ffn_keep_ratio:
            keys_to_prune = {k: 1 for k in importance_measure.keys() if ".mlp.fc" in k}
            ratio = ffn_keep_ratio
        mask = unstructural_pruning(importance_measure, keys_to_prune, ignore_layers, ratio)
    else:
        # use cached indices/masks
        mask = keep_indices_or_masks
    # if is_global:
    #     distilled_vit = t5_modify_global_pruning(distilled_vit, pruned_state_dict)

    pruned_state_dict = {}

    for k, v in vit.state_dict().items():
        pruned_state_dict[k] = v * mask[k].type(v.dtype)

    return pruned_state_dict, mask



def vit_modify_with_weight_init(vit, petl_config, freeze_vit, precision, derivative_info=None, sampled_loader=None, pruned_indices=None, distilled_modify_func=None, importance_measure=None, woodfisher_pruner=None):
    device = list(vit.parameters())[0].device

    vit.to("cpu")

    vit_prune_indices = None

    if petl_config.vit_side_pretrained_weight is not None:
        is_strct_pruning = "unstrct" in petl_config.distillation_init
        num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = petl_config.vit_side_pretrained_weight.split("-")

        num_layers = int(num_layers)
        res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = float(res_keep_ratio), float(attn_keep_ratio), float(ffn_keep_ratio)

        if is_strct_pruning:
            distilled_vit = vit.__class__(
                img_size=vit.img_size,
                patch_size=vit.patch_size,
                use_mean_pooling=False,
                embed_dim=vit.embed_dim,
                attn_dim=vit.attn_dim,
                depth=num_layers,
                num_heads=vit.num_heads,
                num_classes=vit.num_classes,
                mlp_ratio=vit.mlp_ratio,
                qkv_bias=True,
                drop_path_rate=vit.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                use_checkpoint=vit.use_checkpoint,
            )
        else:
            distilled_vit = vit.__class__(
                img_size=vit.img_size,
                patch_size=vit.patch_size,
                use_mean_pooling=False,
                embed_dim=int(vit.embed_dim * res_keep_ratio),
                attn_dim=int(vit.attn_dim * attn_keep_ratio),
                depth=num_layers,
                num_heads=vit.num_heads,
                num_classes=vit.num_classes,
                mlp_ratio=vit.mlp_ratio * ffn_keep_ratio,
                qkv_bias=True,
                drop_path_rate=vit.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                use_checkpoint=vit.use_checkpoint,
            )

        if distilled_modify_func is not None:
            distilled_vit = distilled_modify_func(distilled_vit)

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

            if pruned_indices is not None:
                print("Use pre-extracted pruned indices...")

                if is_strct_pruning:
                    distilled_vit, vit_prune_indices = vit_unstrct_pruning(
                        vit, 
                        distilled_vit, 
                        None,
                        res_keep_ratio, 
                        attn_keep_ratio, 
                        ffn_keep_ratio,
                        is_global="global" in petl_config.distillation_init,
                        pruned_indices=pruned_indices,
                    )
                else:
                    distilled_vit, vit_prune_indices = vit_pruning(
                        vit, 
                        distilled_vit, 
                        None,
                        res_keep_ratio, 
                        attn_keep_ratio, 
                        ffn_keep_ratio,
                        is_global="global" in petl_config.distillation_init,
                        pruned_indices=pruned_indices,
                    )
            elif "prune" in petl_config.distillation_init:
                if derivative_info is not None:
                    for k, v in vit.state_dict().items():
                        if k not in derivative_info:
                            print(k, "not in derivative info")

                if importance_measure is not None:
                    print("Use pre-extracted importance measure.")
                    for k, v in vit.state_dict().items():
                        if k not in importance_measure:
                            print(k, "not in importance_measure")

                elif "mag_prune" in petl_config.distillation_init:
                    # using square of magnitude as the measure.
                    importance_measure = {k: v.type(torch.bfloat16) ** 2 for k, v in vit.state_dict().items()}

                elif "derv_prune" in petl_config.distillation_init:
                    importance_measure = derivative_info

                elif "obs_prune" in petl_config.distillation_init:
                    importance_measure = {k: (v.type(torch.bfloat16) ** 2) * derivative_info[k] for k, v in vit.state_dict().items()}

                elif "zero_prune" in petl_config.distillation_init:
                    importance_measure = {k: torch.zeros_like(v.type(torch.bfloat16)) for k, v in vit.state_dict().items()}

                elif "rand_prune" in petl_config.distillation_init:
                    # (all the importance is random)
                    importance_measure = {k: torch.randn_like(v.type(torch.bfloat16)) for k, v in vit.state_dict().items()}
                elif "woodfisher" in petl_config.distillation_init:
                    importance_measure = derivative_info
                    assert is_strct_pruning == True
                else:
                    raise ValueError("The pruning method is invalid.")

                print(f"Apply {petl_config.distillation_init}...")

                if is_strct_pruning:
                    distilled_vit, vit_prune_indices = vit_unstrct_pruning(
                        vit, 
                        distilled_vit, 
                        importance_measure,
                        res_keep_ratio, 
                        attn_keep_ratio, 
                        ffn_keep_ratio,
                        is_global="global" in petl_config.distillation_init,
                        pruned_indices=None,
                    )

                    if "woodfisher" in petl_config.distillation_init:
                        pruned_weights = woodfisher_pruner.reweighting_after_pruning(
                            vit.state_dict(), vit_prune_indices
                        )

                        distilled_vit.load_state_dict(pruned_weights)
                else:
                    distilled_vit, vit_prune_indices = vit_pruning(
                        vit, 
                        distilled_vit, 
                        importance_measure,
                        res_keep_ratio, 
                        attn_keep_ratio, 
                        ffn_keep_ratio,
                        is_global="global" in petl_config.distillation_init,
                        pruned_indices=None,
                    )

            if "fusion" in petl_config.distillation_init:
                print("Apply fusion on ViT...")
                distilled_vit = vit_fusion(
                    vit, 
                    distilled_vit, 
                    distill_merge_ratio=petl_config.distill_merge_ratio, 
                    exact=petl_config.exact, 
                    normalization=petl_config.normalization, 
                    metric=petl_config.metric,
                    to_one=petl_config.to_one,
                    importance=petl_config.importance,
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
        torch.cuda.empty_cache()
        
        return distilled_vit, vit_prune_indices, importance_measure
    
    vit.to(device)

    return vit, vit_prune_indices, importance_measure


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    torch.manual_seed(0)


    class Config:
        vit_side_pretrained_weight = "39-1.0-0.5-0.5"
        distillation_init = "mag_prune"
        distill_merge_ratio = 0.5
        exact = True
        normalization = False
        metric = "dot"
        to_one = False
        importance = False

    config = Config()

    x = torch.randn(2, 3, 224, 224)
    vit, _ = Blip2Base.init_vision_encoder(
        "eva_clip_g", 224, 0, False, "fp32"
    )

    vit.eval()

    old_output = vit(x)

    derivative_info = {k: v ** 2 for k, v in vit.state_dict().items()}

    vit, _ = vit_modify_with_weight_init(vit, config, True, "fp32", derivative_info)

    new_output = vit(x)

    print("diff: ", torch.mean((old_output - new_output) ** 2))
        