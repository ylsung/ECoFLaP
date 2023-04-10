

# Codes modified from https://github.com/samuela/git-re-basin and https://github.com/themrzmaster/git-re-basin-pytorch

import re
import random
import torch
import torch.nn as nn
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

from typing import NamedTuple


def weight_matching(ps, params_a, params_b, max_iter=100, init_perm=None, silent=True):
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

    if not silent:
        print(perm_sizes)

    perm = {p: torch.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = torch.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))

                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()
            oldL = torch.einsum('ij,ij->i', A, torch.eye(n)[perm[p].long()]).sum()
            newL = torch.einsum('ij,ij->i', A,torch.eye(n)[ci, :]).sum()

            if not silent:
                print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = torch.Tensor(ci)

        if not progress:
            break

    return perm


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict):
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)

def get_permuted_param(ps, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):

        # print("in")
        # print(except_axis, axis, p, k)
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            w = torch.index_select(w, axis, perm[p].int())

    return w

def apply_permutation(ps, perm, params):
    """Apply a `perm` to `params`."""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def mlp_permutation_spec(num_hidden_layers: int):
    """We assume that one permutation cannot appear in two axes of the same weight array."""
    assert num_hidden_layers >= 1
    return permutation_spec_from_axes_to_perm({
        "layer0.weight": ("P_0", None),
        **{f"layer{i}.weight": ( f"P_{i}", f"P_{i-1}")
            for i in range(1, num_hidden_layers)},
        **{f"layer{i}.bias": (f"P_{i}", )
            for i in range(num_hidden_layers)},
        f"layer{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
        f"layer{num_hidden_layers}.bias": (None, ),
    })

def test_weight_matching():
    """If we just have a single hidden layer then it should converge after just one step."""
    ps = mlp_permutation_spec(num_hidden_layers=1)
    print(ps.axes_to_perm)
    rng = torch.Generator()
    rng.manual_seed(13)
    num_hidden = 10
    shapes = {
        "layer0.weight": (num_hidden, 2),
        "layer0.bias": (num_hidden, ),
        "layer1.weight": (3, num_hidden),
        "layer1.bias": (3, )
    }

    params_a = {k: torch.randn(shape, generator=rng) for k, shape in shapes.items()}
    params_b = {k: torch.randn(shape, generator=rng) for k, shape in shapes.items()}

    t = torch.randperm(num_hidden)
    params_b["layer0.weight"] = params_a["layer0.weight"][t, :]
    params_b["layer0.bias"] = params_a["layer0.bias"][t]
    params_b["layer1.weight"] = params_a["layer1.weight"][:, t]
    params_b["layer1.bias"] = deepcopy(params_a["layer1.bias"])

    perm = weight_matching(ps, params_a, params_b)

    updated_params = apply_permutation(ps, perm, params_b)
    print(perm)
    print(params_a)
    # print(params_b)
    print(updated_params)


def permute_based_on_first_layer(transformer):
    state_dict = transformer.state_dict()

    encoder_layers = transformer.config.num_layers
    decoder_layers = transformer.config.num_decoder_layers
    num_heads = transformer.config.num_heads

    layers_with_heads = {
        "encoder": [
            "encoder.block.0.layer.0.SelfAttention.q.weight",
            "encoder.block.0.layer.0.SelfAttention.k.weight",
            "encoder.block.0.layer.0.SelfAttention.v.weight",
            "encoder.block.0.layer.0.SelfAttention.o.weight",
        ],
        "decoder": [
            "decoder.block.0.layer.0.SelfAttention.q.weight",
            "decoder.block.0.layer.0.SelfAttention.k.weight",
            "decoder.block.0.layer.0.SelfAttention.v.weight",
            "decoder.block.0.layer.0.SelfAttention.o.weight",
            "decoder.block.0.layer.1.EncDecAttention.q.weight",
            "decoder.block.0.layer.1.EncDecAttention.k.weight",
            "decoder.block.0.layer.1.EncDecAttention.v.weight",
            "decoder.block.0.layer.1.EncDecAttention.o.weight",
        ]
    }

    encoder_ps = permutation_spec_from_axes_to_perm({
        **{f"encoder.block.0.layer.0.SelfAttention.q.weight.{i}": (f"P_self_qk_{i}", None) for i in range(num_heads)},                                                                                                 
        **{f"encoder.block.0.layer.0.SelfAttention.k.weight.{i}": (f"P_self_qk_{i}", None) for i in range(num_heads)},                                                                                                 
        **{f"encoder.block.0.layer.0.SelfAttention.v.weight.{i}": (f"P_self_vo_{i}", None) for i in range(num_heads)},                                                                                                 
        **{f"encoder.block.0.layer.0.SelfAttention.o.weight.{i}": (None, f"P_self_vo_{i}") for i in range(num_heads)},                                                                                                 
        **{"encoder.block.0.layer.0.layer_norm.weight": (None,)},                                                                                                      
        **{"encoder.block.0.layer.1.DenseReluDense.wi_0.weight": ("P_ffn", None)},
        **{"encoder.block.0.layer.1.DenseReluDense.wi_1.weight": ("P_ffn", None)},                                                                                                
        **{"encoder.block.0.layer.1.DenseReluDense.wo.weight": (None, "P_ffn")},                                                                                               
        **{"encoder.block.0.layer.1.layer_norm.weight": (None,)},
    })

    decoder_ps = permutation_spec_from_axes_to_perm({
        **{f"decoder.block.0.layer.0.SelfAttention.q.weight.{i}": (f"P_self_qk_{i}", None) for i in range(num_heads)},                                                                                                 
        **{f"decoder.block.0.layer.0.SelfAttention.k.weight.{i}": (f"P_self_qk_{i}", None) for i in range(num_heads)},                                                                                                 
        **{f"decoder.block.0.layer.0.SelfAttention.v.weight.{i}": (f"P_self_vo_{i}", None) for i in range(num_heads)},                                                                                                 
        **{f"decoder.block.0.layer.0.SelfAttention.o.weight.{i}": (None, f"P_self_vo_{i}") for i in range(num_heads)},                                                                                                 
        **{"decoder.block.0.layer.0.layer_norm.weight": (None,)},                                                                                                      
        **{f"decoder.block.0.layer.1.EncDecAttention.q.weight.{i}": (f"P_cross_qk_{i}", None) for i in range(num_heads)},                                                                                               
        **{f"decoder.block.0.layer.1.EncDecAttention.k.weight.{i}": (f"P_cross_qk_{i}", None) for i in range(num_heads)},                                                                                               
        **{f"decoder.block.0.layer.1.EncDecAttention.v.weight.{i}": (f"P_cross_vo_{i}", None) for i in range(num_heads)},                                                                                               
        **{f"decoder.block.0.layer.1.EncDecAttention.o.weight.{i}": (None, f"P_cross_vo_{i}") for i in range(num_heads)},                                                                                               
        **{"decoder.block.0.layer.1.layer_norm.weight": (None,)},                                                                                                      
        **{"decoder.block.0.layer.2.DenseReluDense.wi_0.weight": ("P_ffn", None)},   
        **{"decoder.block.0.layer.2.DenseReluDense.wi_1.weight": ("P_ffn", None)},                                                                                                
        **{"decoder.block.0.layer.2.DenseReluDense.wo.weight": (None, "P_ffn")},                                                                                               
        **{"decoder.block.0.layer.2.layer_norm.weight": (None,)},
    })

    permuted_layers = []

    permuted_weights = deepcopy(state_dict) # the first block is initialize here and others will overwrite with the permuted version later

    # split weights for num_heads

    state_dict_with_split_heads = {}

    for k, v in state_dict.items():
        if "Attention" in k:
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

    for transformer_type, num_layers, ps in zip(["encoder", "decoder"], [encoder_layers, decoder_layers], [encoder_ps, decoder_ps]):
        layers_as_base = {}

        for k, v in state_dict.items():
            # take the first block as the base
            if f"{transformer_type}.block.0" in k and "relative_attention_bias" not in k:
                layers_as_base[k] = v

        for i in range(1, num_layers):
            layers_to_permute = {}

            for k in layers_as_base.keys():
                new_k = k.replace("block.0", f"block.{i}")

                layers_to_permute[k] = state_dict[new_k] # need to use the same name for the permutation functions

            perm = weight_matching(ps, layers_as_base, layers_to_permute, silent=True)

            # print(perm)

            # print(layers_to_permute.keys())

            layers_to_permute = apply_permutation(ps, perm, layers_to_permute)

            # combine the weights of attention
            for layer_with_heads in layers_with_heads[transformer_type]:
                weights = []
                for head_idx in range(num_heads):
                    layer_with_heads_this_head = layer_with_heads + f".{head_idx}"
                    weights.append(layers_to_permute[layer_with_heads_this_head])

                new_layer_with_heads = layer_with_heads.replace("block.0", f"block.{i}")

                if layer_with_heads.endswith("o.weight"):
                    permuted_weights[new_layer_with_heads] = torch.cat(weights, dim=1)
                else:
                    permuted_weights[new_layer_with_heads] = torch.cat(weights, dim=0)

                permuted_layers.append(new_layer_with_heads)

                for head_idx in range(num_heads):
                    layer_with_heads_this_head = layer_with_heads + f".{head_idx}"

                    del layers_to_permute[layer_with_heads_this_head]

            for k, v in layers_to_permute.items():
                new_k = k.replace("block.0", f"block.{i}")

                # print(new_k)

                permuted_layers.append(new_k)

                permuted_weights[new_k] = v

    # for k, v in permuted_weights.items():
    #     if k not in permuted_layers:
    #         print(k)

    transformer.load_state_dict(permuted_weights)

    return transformer


def permute_based_on_block(transformer, distilled_block_ids):
    state_dict = transformer.state_dict()

    encoder_layers = transformer.config.num_layers
    decoder_layers = transformer.config.num_decoder_layers
    num_heads = transformer.config.num_heads

    layers_with_heads = {
        "encoder": [
            "encoder.block.0.layer.0.SelfAttention.q.weight",
            "encoder.block.0.layer.0.SelfAttention.k.weight",
            "encoder.block.0.layer.0.SelfAttention.v.weight",
            "encoder.block.0.layer.0.SelfAttention.o.weight",
        ],
        "decoder": [
            "decoder.block.0.layer.0.SelfAttention.q.weight",
            "decoder.block.0.layer.0.SelfAttention.k.weight",
            "decoder.block.0.layer.0.SelfAttention.v.weight",
            "decoder.block.0.layer.0.SelfAttention.o.weight",
            "decoder.block.0.layer.1.EncDecAttention.q.weight",
            "decoder.block.0.layer.1.EncDecAttention.k.weight",
            "decoder.block.0.layer.1.EncDecAttention.v.weight",
            "decoder.block.0.layer.1.EncDecAttention.o.weight",
        ]
    }

    encoder_ps = permutation_spec_from_axes_to_perm({
        **{f"encoder.block.0.layer.0.SelfAttention.q.weight.{i}": (f"P_self_qk_{i}", None) for i in range(num_heads)},                                                                                                 
        **{f"encoder.block.0.layer.0.SelfAttention.k.weight.{i}": (f"P_self_qk_{i}", None) for i in range(num_heads)},                                                                                                 
        **{f"encoder.block.0.layer.0.SelfAttention.v.weight.{i}": (f"P_self_vo_{i}", None) for i in range(num_heads)},                                                                                                 
        **{f"encoder.block.0.layer.0.SelfAttention.o.weight.{i}": (None, f"P_self_vo_{i}") for i in range(num_heads)},                                                                                                 
        **{"encoder.block.0.layer.0.layer_norm.weight": (None,)},                                                                                                      
        **{"encoder.block.0.layer.1.DenseReluDense.wi_0.weight": ("P_ffn", None)},
        **{"encoder.block.0.layer.1.DenseReluDense.wi_1.weight": ("P_ffn", None)},                                                                                              
        **{"encoder.block.0.layer.1.DenseReluDense.wo.weight": (None, "P_ffn")},                                                                                               
        **{"encoder.block.0.layer.1.layer_norm.weight": (None,)},
    })

    decoder_ps = permutation_spec_from_axes_to_perm({
        **{f"decoder.block.0.layer.0.SelfAttention.q.weight.{i}": (f"P_self_qk_{i}", None) for i in range(num_heads)},                                                                                                 
        **{f"decoder.block.0.layer.0.SelfAttention.k.weight.{i}": (f"P_self_qk_{i}", None) for i in range(num_heads)},                                                                                                 
        **{f"decoder.block.0.layer.0.SelfAttention.v.weight.{i}": (f"P_self_vo_{i}", None) for i in range(num_heads)},                                                                                                 
        **{f"decoder.block.0.layer.0.SelfAttention.o.weight.{i}": (None, f"P_self_vo_{i}") for i in range(num_heads)},                                                                                                 
        **{"decoder.block.0.layer.0.layer_norm.weight": (None,)},                                                                                                      
        **{f"decoder.block.0.layer.1.EncDecAttention.q.weight.{i}": (f"P_cross_qk_{i}", None) for i in range(num_heads)},                                                                                               
        **{f"decoder.block.0.layer.1.EncDecAttention.k.weight.{i}": (f"P_cross_qk_{i}", None) for i in range(num_heads)},                                                                                               
        **{f"decoder.block.0.layer.1.EncDecAttention.v.weight.{i}": (f"P_cross_vo_{i}", None) for i in range(num_heads)},                                                                                               
        **{f"decoder.block.0.layer.1.EncDecAttention.o.weight.{i}": (None, f"P_cross_vo_{i}") for i in range(num_heads)},                                                                                               
        **{"decoder.block.0.layer.1.layer_norm.weight": (None,)},                                                                                                      
        **{"decoder.block.0.layer.2.DenseReluDense.wi_0.weight": ("P_ffn", None)},      
        **{"decoder.block.0.layer.2.DenseReluDense.wi_1.weight": ("P_ffn", None)},                                                                                             
        **{"decoder.block.0.layer.2.DenseReluDense.wo.weight": (None, "P_ffn")},                                                                                               
        **{"decoder.block.0.layer.2.layer_norm.weight": (None,)},
    })

    permuted_layers = []

    permuted_weights = deepcopy(state_dict) # will overwrite with permuted version later

    # split weights for num_heads

    state_dict_with_split_heads = {}

    for k, v in state_dict.items():
        if "Attention" in k:
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

    for transformer_type, num_layers, ps in zip(["encoder", "decoder"], [encoder_layers, decoder_layers], [encoder_ps, decoder_ps]):
        
        for block_ids_to_distill in distilled_block_ids:
            layers_as_base = {}
            if isinstance(block_ids_to_distill, int):
                block_ids_to_distill = [block_ids_to_distill]

            block_ids_to_distill = list(block_ids_to_distill)

            base_id = max(len(block_ids_to_distill) - 1, 0) // 2 # determine which block in the list is the base

            base_block_id = block_ids_to_distill[base_id] # get the index that corresponds to which transformer layer

            block_ids_to_distill.pop(base_id)

            for k, v in state_dict.items():
                # take the block as the base
                if re.match(rf"{transformer_type}.block.{base_block_id}\b.*", k) and "relative_attention_bias" not in k:
                # if f"{transformer_type}.block.{base_block_id}" in k and "relative_attention_bias" not in k:
                    new_k = k.replace(f"block.{base_block_id}", f"block.0") # change name to the name with block.0
                    layers_as_base[new_k] = v

            if len(block_ids_to_distill) == 0:
                # there is no thing to merge, and the base layer is already in the permuted_state_dict
                continue

            for i in block_ids_to_distill:
                layers_to_permute = {}

                for k in layers_as_base.keys():
                    new_k = k.replace(f"block.0", f"block.{i}")

                    # print(k, new_k)

                    layers_to_permute[k] = state_dict[new_k] # need to use the same name for the permutation functions

                perm = weight_matching(ps, layers_as_base, layers_to_permute, silent=True)

                # print(perm)

                # print(layers_to_permute.keys())

                layers_to_permute = apply_permutation(ps, perm, layers_to_permute)

                # combine the weights of attention
                for layer_with_heads in layers_with_heads[transformer_type]:
                    weights = []
                    for head_idx in range(num_heads):
                        layer_with_heads_this_head = layer_with_heads + f".{head_idx}"
                        weights.append(layers_to_permute[layer_with_heads_this_head])

                    new_layer_with_heads = layer_with_heads.replace("block.0", f"block.{i}")

                    if layer_with_heads.endswith("o.weight"):
                        permuted_weights[new_layer_with_heads] = torch.cat(weights, dim=1)
                    else:
                        permuted_weights[new_layer_with_heads] = torch.cat(weights, dim=0)

                    permuted_layers.append(new_layer_with_heads)

                    for head_idx in range(num_heads):
                        layer_with_heads_this_head = layer_with_heads + f".{head_idx}"

                        del layers_to_permute[layer_with_heads_this_head]

                for k, v in layers_to_permute.items():
                    new_k = k.replace("block.0", f"block.{i}")

                    # print(new_k)

                    permuted_layers.append(new_k)

                    permuted_weights[new_k] = v

    # for k, v in permuted_weights.items():
    #     if k not in permuted_layers:
    #         print(k)

    transformer.load_state_dict(permuted_weights)

    return transformer


if __name__ == "__main__":
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import argparse

    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--adapter_type",
        default="normal",
        type=str,
        choices=["normal", "lowrank", "compacter"],
    )
    args = parser.parse_args()

    class AdapterConfig:
        def __init__(self, adapter_type):
            self.adapter_type = adapter_type

            # Adapter Config
            self.pbp_reduction_factor = 1
            self.trainable_param_names = ".*merge_weights.*"
            self.enc_num_tokens = 0
            self.dec_num_tokens = 0
            self.prompts_expand_after = True
            self.hard_prompt = None # "Find the relationship between the two concatenated sentences, or classify it if there is only one sentence."
            self.init_from_emb = True

            self.side_pretrained_weight = "6-768"
            self.distillation_init = "sum"
            self.distilled_block_ids = "[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]"
            self.distilled_block_weights = None
            self.rep_stack_forward = True
            self.scaling_factor = 1.0
            self.learnable_weight_type = "scalar-shared"
            self.modules_to_merge = ".*layer_norm.*|.*DenseReluDense.*"

    config = AdapterConfig(args.adapter_type)
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    # input_seq = tokenizer(
    #     ["Applies a linear transformation to the incoming data.", "Parameters: in_features - size of each input sample. out_features - size of each output sample."],
    #     return_tensors="pt",
    #     truncation=True,
    #     padding=True,
    # )
    # target_seq = tokenizer(
    #     ["Parameters: in_features - size of each input sample. out_features - size of each output sample.", "Applies a linear transformation to the incoming data."],
    #     return_tensors="pt",
    #     truncation=True,
    #     padding=True,
    # )

    input_seq = tokenizer(
        ["Applies a linear transformation to the incoming data."],
        return_tensors="pt",
    )
    target_seq = tokenizer(
        ["Parameters: in_features - size of each input sample. out_features - size of each output sample."],
        return_tensors="pt",
    )

    print("Input shape: ", input_seq.input_ids.shape)
    print("Target shape: ", target_seq.input_ids[:, 1:].shape)

    print("Old model")
    # print(model)
    # print(model.state_dict().keys())
    old_param = model.state_dict()

    model.eval()
    with torch.no_grad():
        old_outputs = model(
            input_ids=input_seq.input_ids,
            decoder_input_ids=target_seq.input_ids[:, :-1],
            labels=target_seq.input_ids[:, 1:],
        )

    generation_input_ids = tokenizer(["translate English to German: The house is wonderful."], return_tensors="pt")
    old_generation_outputs = model.generate(**generation_input_ids, num_beams=1)

    permuted_model = permute_based_on_first_layer(model)

    permuted_model.eval()
    with torch.no_grad():
        new_outputs = permuted_model(
            input_ids=input_seq.input_ids,
            decoder_input_ids=target_seq.input_ids[:, :-1],
            labels=target_seq.input_ids[:, 1:],
        )

    print(f"Logits diff {torch.abs(old_outputs.logits - new_outputs.logits).mean():.3f}")
    print(f"Loss diff old={old_outputs.loss:.3f} new={new_outputs.loss:.3f}")

    new_generation_outputs = permuted_model.generate(**generation_input_ids, num_beams=1)

    print(old_generation_outputs)
    print(new_generation_outputs)

    print(tokenizer.batch_decode(old_generation_outputs))
    print(tokenizer.batch_decode(new_generation_outputs))
