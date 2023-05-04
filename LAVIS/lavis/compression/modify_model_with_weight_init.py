import torch
import torch.nn as nn

import re
from copy import deepcopy
import functools

import transformers
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from typing import Optional, Tuple, Union, Dict, Any

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, BaseModelOutput

from transformers import AutoTokenizer

from collections import defaultdict

from lavis.compression.learnable_merge import t5_modify_for_learnable_merge, convert_to_normal_save_weights
from lavis.compression.weight_matching import permute_based_on_first_layer, permute_based_on_block
from lavis.compression.structural_pruning import pruning, get_pruned_dim, fusion
from lavis.compression.unstructural_pruning import t5_unstrct_pruning

def get_activations(transformer, sampled_loader, distilled_block_ids, cache_type="representations"):
    assert cache_type in ["representations", "input_gram", "input_output_gram"]

    input_representations = defaultdict(list) if cache_type == "representations" else defaultdict(float)
    output_representations = defaultdict(list) if cache_type == "representations" else defaultdict(float)

    inter_artifacts = defaultdict(float)

    def hook_input_output(module, input, output):
        # module_name should be a dictionary
        if getattr(module, "bias", None) is not None:
            output_remove_bias = output - module.bias
        else:
            output_remove_bias = output

        input_representations[module.module_name].append(input[0].permute(1, 0, 2).detach().cpu())
        output_representations[module.module_name].append(output_remove_bias.permute(1, 0, 2).detach().cpu())

    def hook_input_output_attn(module, input, output):
        # module_name should be a dictionary
        qkv_inputs = input[0]

        if getattr(module, "bias", None) is not None:
            output_remove_bias = output - module.bias
        else:
            output_remove_bias = output
            
        input_representations[module.module_name].append(qkv_inputs.permute(1, 0, 2).detach().cpu())
        output_representations[module.module_name].append(output_remove_bias.permute(1, 0, 2).detach().cpu())

    def hook_input_gram(module, input, output):
        # module_name should be a dictionary
        input = input[0].permute(1, 0, 2)

        flatten_input = input.reshape(-1, input.shape[-1]) # (B * L, D)
        gram = torch.matmul(flatten_input.T, flatten_input)

        input_representations[module.module_name] += gram.detach().cpu()

    def hook_input_gram_attn(module, input, output):
        # module_name should be a dictionary
        # tuple, tensor
        qkv_inputs = input[0].permute(1, 0, 2)

        flatten_input = qkv_inputs.reshape(-1, qkv_inputs.shape[-1]) # (B * L, D)
        gram = torch.matmul(flatten_input.T, flatten_input)

        input_representations[module.module_name] += gram.detach().cpu()

    def hook_input_output_gram(module, input, output):
        # module_name should be a dictionary
        input = input[0].permute(1, 0, 2)

        flatten_input = input.reshape(-1, input.shape[-1]) # (B * L, D)
        gram = torch.matmul(flatten_input.T, flatten_input)

        input_representations[module.module_name] += gram.detach().cpu()

        cur_module_block = getattr(module, "cur_module_block", None)
        pre_module_block = getattr(module, "pre_module_block", None)

        if cur_module_block != pre_module_block:
            if int(cur_module_block.split(".")[-1]) > int(pre_module_block.split(".")[-1]):
                # print(module.module_name, "input_output")
                # later block will use previous reprensentation for computations.
                flatten_output = output.reshape(-1, output.shape[-1]).detach().cpu()
                pre_name = module.module_name.replace(cur_module_block, pre_module_block)
                pre_input = inter_artifacts[pre_name]
                flatten_pre_input = pre_input.reshape(-1, pre_input.shape[-1])
                gram = torch.matmul(flatten_output.T, flatten_pre_input)
                output_representations[module.module_name] += gram.detach().cpu()
                # erase the used artifacts
                del inter_artifacts[pre_name]
            else:
                # print(module.module_name, "cache for input_output")
                # store the activation that will be used later
                inter_artifacts[module.module_name] = input.detach().cpu() # assign, not sum


    def hook_input_output_gram_attn(module, input, output):
        # module_name should be a dictionary
        # tuple, tensor
        input = input[0].permute(1, 0, 2)

        flatten_input = input.reshape(-1, input.shape[-1]) # (B * L, D)
        gram = torch.matmul(flatten_input.T, flatten_input)

        input_representations[module.module_name] += gram.detach().cpu()

        cur_module_block = getattr(module, "cur_module_block", None)
        pre_module_block = getattr(module, "pre_module_block", None)

        if cur_module_block != pre_module_block:
            if int(cur_module_block.split(".")[-1]) > int(pre_module_block.split(".")[-1]):
                # print(module.module_name, "input_output")
                # later block will use previous reprensentation for computations.
                flatten_output = output.reshape(-1, output.shape[-1]).detach().cpu()
                pre_name = module.module_name.replace(cur_module_block, pre_module_block)
                pre_input = inter_artifacts[pre_name]
                flatten_pre_input = pre_input.reshape(-1, pre_input.shape[-1])
                gram = torch.matmul(flatten_output.T, flatten_pre_input)
                output_representations[module.module_name] += gram.detach().cpu()
                # erase the used artifacts
                del inter_artifacts[pre_name]
            else:
                # print(module.module_name, "cache for input_output")
                # store the activation that will be used later
                inter_artifacts[module.module_name] = input.detach().cpu() # assign, not sum

    if cache_type == "input_gram":
        hook_others = hook_input_gram
        hook_attn = hook_input_gram_attn
    elif cache_type == "representations":
        hook_others = hook_input_output
        hook_attn = hook_input_output_attn
    elif cache_type == "input_output_gram":
        hook_others = hook_input_output_gram
        hook_attn = hook_input_output_gram_attn

    keys_to_track = [
        "SelfAttention.q",
        "SelfAttention.k",
        "SelfAttention.v",
        "SelfAttention.o",
        "EncDecAttention.q",
        "EncDecAttention.k",
        "EncDecAttention.v",
        "EncDecAttention.o",
        "DenseReluDense.wi",
        "DenseReluDense.wo",
    ]

    hooks = []

    if distilled_block_ids is not None:
        block_to_add_input_output_gram = {}

        for block in distilled_block_ids:
            if isinstance(block, list) and len(block) > 1:
                for b in block:
                    # add input gram
                    block_to_add_input_output_gram[f"block.{b}"] = f"block.{b}"

                # overwrite if the positions are used to compute input output gram
                block_to_add_input_output_gram[f"block.{block[-1]}"] = f"block.{block[0]}"
                block_to_add_input_output_gram[f"block.{block[0]}"] = f"block.{block[-1]}"

    else:
        block_to_add_input_output_gram = None

    def return_the_matched_modules(name):
        if block_to_add_input_output_gram == None:
            return True, None, None
        else:
            for k, v in block_to_add_input_output_gram.items():
                if re.fullmatch(rf".*\b{k}\b.*", name):
                    return True, k, v
            else:
                return False, None, None


    for name, module in transformer.named_modules():
        if any([name.endswith(n) for n in keys_to_track]):
            module.module_name = name

            need_hook, cur_module_block, pre_module_block = return_the_matched_modules(name)

            module.cur_module_block = cur_module_block
            module.pre_module_block = pre_module_block

            if need_hook:
                if "Attention" in name:
                    hook = module.register_forward_hook(hook_attn)
                else:
                    hook = module.register_forward_hook(hook_others)
                hooks.append(hook)

    transformer.float()
    transformer.eval()
    transformer.to("cuda")

    for inputs in sampled_loader:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to("cuda")
        feature = transformer(**inputs)

    transformer.to("cpu")

    if cache_type == "representations":
        for k, v in input_representations.items():
            input_representations[k] = torch.cat(v, dim=0)

        for k, v in output_representations.items():
            output_representations[k] = torch.cat(v, dim=0)

    # for k, v in input_representations.items():
    #     print(v.shape)

    # remove hooks
    for h in hooks:
        h.remove()

    return input_representations, output_representations


def get_layer_ids(merge_block):
    ids = []
    now_id = 0
    for m in merge_block:
        m = [m] if isinstance(m, int) else m
        for _ in m:
            ids.append(now_id)
        now_id += 1
    return ids


def rep_stack_forward(
    self,
    input_ids=None,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    inputs_embeds=None,
    head_mask=None,
    cross_attn_head_mask=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    # Model parallel
    if self.model_parallel:
        torch.cuda.set_device(self.first_device)
        self.embed_tokens = self.embed_tokens.to(self.first_device)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        err_msg_prefix = "decoder_" if self.is_decoder else ""
        raise ValueError(
            f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        err_msg_prefix = "decoder_" if self.is_decoder else ""
        raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

    if inputs_embeds is None:
        assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = input_shape

    # required mask seq length can be calculated via length of past
    mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

    if use_cache is True:
        assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

    if attention_mask is None:
        attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
    if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
        encoder_seq_length = encoder_hidden_states.shape[1]
        encoder_attention_mask = torch.ones(
            batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
        )

    # initialize past_key_values with `None` if past does not exist
    if past_key_values is None:
        past_key_values = [None] * len(self.layer_ids_list)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    head_mask = self.get_head_mask(head_mask, len(self.layer_ids_list))
    cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, len(self.layer_ids_list))
    present_key_value_states = () if use_cache else None
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and self.is_decoder) else None
    position_bias = None
    encoder_decoder_position_bias = None

    hidden_states = self.dropout(inputs_embeds)

    for real_i, i in enumerate(self.layer_ids_list):
        layer_module = self.block[i]
        past_key_value = past_key_values[real_i]
        layer_head_mask = head_mask[real_i]
        cross_attn_layer_head_mask = cross_attn_head_mask[real_i]
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if position_bias is not None:
                position_bias = position_bias.to(hidden_states.device)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
            if encoder_extended_attention_mask is not None:
                encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
            if encoder_decoder_position_bias is not None:
                encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
            if layer_head_mask is not None:
                layer_head_mask = layer_head_mask.to(hidden_states.device)
            if cross_attn_layer_head_mask is not None:
                cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return tuple(module(*inputs, use_cache, output_attentions))

                return custom_forward

            layer_outputs = checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                extended_attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,
                layer_head_mask,
                cross_attn_layer_head_mask,
                None,  # past_key_value is always None with gradient checkpointing
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        # layer_outputs is a tuple with:
        # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
        if use_cache is False:
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

        hidden_states, present_key_value_state = layer_outputs[:2]

        # We share the position biases between the layers - the first layer store them
        # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
        # (cross-attention position bias), (cross-attention weights)
        position_bias = layer_outputs[2]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
        # append next layer key value states
        if use_cache:
            present_key_value_states = present_key_value_states + (present_key_value_state,)

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[3],)
            if self.is_decoder:
                all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    # Add last layer
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                present_key_value_states,
                all_hidden_states,
                all_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=present_key_value_states,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
        cross_attentions=all_cross_attentions,
    )


def get_uniform_weights(total_len):
    return [1 / total_len for _ in range(total_len)]


def weights_interpolation_from_different_layers(weights_to_distill, target_names, distilled_block_ids, distilled_block_weights, modules_to_merge):
    weights = {}

    for k in target_names:
        current_block = re.findall("block[.][0-9]*", k)
        assert len(current_block) <= 1
        current_block = current_block[0] if len(current_block) > 0 else None

        if current_block is None or "relative_attention_bias" in k:
            weights[k] = weights_to_distill[k]
            continue

        current_block_id = int(current_block.split(".")[-1])

        block_ids_to_distill = distilled_block_ids[current_block_id]

        if not isinstance(block_ids_to_distill, list):
            block_ids_to_distill = [block_ids_to_distill]

        block_ids_to_distill = list(block_ids_to_distill)

        if distilled_block_weights is None:
            # evenly distributed weights
            print(len(block_ids_to_distill))
            block_weights_to_distill = get_uniform_weights(len(block_ids_to_distill))
        else:
            block_weights_to_distill = distilled_block_weights[current_block_id]

        if not isinstance(block_weights_to_distill, list):
            block_weights_to_distill = [block_weights_to_distill]

        weight_to_merge = 0

        print(block_ids_to_distill, block_weights_to_distill)

        assert len(block_ids_to_distill) == len(block_weights_to_distill)

        if re.fullmatch(modules_to_merge, k):
            # merge
            for block_id, block_weight in zip(block_ids_to_distill, block_weights_to_distill):
                merge_block = f"block.{block_id}"

                merge_k = k.replace(current_block, merge_block)

                # print("Merge: ", merge_k)

                weight_to_merge = weight_to_merge + weights_to_distill[merge_k] * block_weight

        else:
            # just take the first weight in the merge block
            merge_block = f"block.{block_ids_to_distill[0]}"

            merge_k = k.replace(current_block, merge_block)

            # print("Don't merge: ", merge_k)

            weight_to_merge = weights_to_distill[merge_k]

        weights[k] = weight_to_merge

    return weights


def rescale_gram_matrice(gram, scaling_factor):
    scaling_for_non_diag = scaling_factor
    diag = torch.diag_embed(torch.diag(gram))

    return scaling_for_non_diag * gram + (1 - scaling_for_non_diag) * diag


def weights_regmean_from_different_layers(weights_to_distill, target_names, distilled_block_ids, representations, scaling_factor):
    input_representations, output_representations = representations
    weights = {}

    for k in target_names:
        current_block = re.findall("block[.][0-9]*", k)
        assert len(current_block) <= 1
        current_block = current_block[0] if len(current_block) > 0 else None

        if current_block is None or "relative_attention_bias" in k:
            weights[k] = weights_to_distill[k]
            continue

        current_block_id = int(current_block.split(".")[-1])

        block_ids_to_distill = distilled_block_ids[current_block_id]

        if not isinstance(block_ids_to_distill, list):
            block_ids_to_distill = [block_ids_to_distill]

        block_ids_to_distill = list(block_ids_to_distill)

        weight_to_merge = 0

        print(current_block_id, block_ids_to_distill)

        total_gram = 0

        for block_id in block_ids_to_distill:
            merge_block = f"block.{block_id}"

            merge_k = k.replace(current_block, merge_block)

            if len(block_ids_to_distill) == 1:
                weight_to_merge = weights_to_distill[merge_k]
            else:
                # regmean for weight
                if "weight" in merge_k and ".layer_norm" not in merge_k:
                    input_rep_k = merge_k[:-7]
                    gram = input_representations[input_rep_k]
    
                    gram = rescale_gram_matrice(gram, scaling_factor)

                    total_gram = total_gram + gram

                    # print("regmean: ", merge_k)

                    weight_to_merge = weight_to_merge + torch.matmul(weights_to_distill[merge_k], gram)
                else: # simple average for bias and ln
                    # print("direct merge: ", merge_k)
                    weight_to_merge = weight_to_merge + weights_to_distill[merge_k] / len(block_ids_to_distill)

        if not isinstance(total_gram, int): # weight
            # torch.linalg.pinv
            # print(k)
            # normalization_factor = torch.linalg.pinv(total_gram, hermitian=True)
            # weight_to_merge = torch.matmul(weight_to_merge, normalization_factor)

            weight_to_merge = torch.linalg.lstsq(total_gram.T, weight_to_merge.T).solution.T

        # if not torch.all(torch.isclose(weight_to_merge, weights_to_distill[merge_k])):
        #     print(merge_k, (weight_to_merge - weights_to_distill[merge_k]).abs().mean())

        weights[k] = weight_to_merge

    return weights


def weights_regmean_distill_from_different_layers(weights_to_distill, target_names, distilled_block_ids, representations, scaling_factor):
    input_representations, output_representations = representations
    weights = {}

    for k in target_names:
        current_block = re.findall("block[.][0-9]*", k)
        assert len(current_block) <= 1
        current_block = current_block[0] if len(current_block) > 0 else None

        if current_block is None or "relative_attention_bias" in k:
            weights[k] = weights_to_distill[k]
            continue

        current_block_id = int(current_block.split(".")[-1])

        block_ids_to_distill = distilled_block_ids[current_block_id]

        if not isinstance(block_ids_to_distill, list):
            block_ids_to_distill = [block_ids_to_distill]

        block_ids_to_distill = list(block_ids_to_distill)

        weight_to_merge = 0

        print(current_block_id, block_ids_to_distill)

        total_gram = 0

        for block_id in block_ids_to_distill:
            merge_block = f"block.{block_id}"

            merge_k = k.replace(current_block, merge_block)

            if len(block_ids_to_distill) == 1:
                weight_to_merge = weights_to_distill[merge_k]
            else:
                # regmean for weight
                if "weight" in merge_k and ".layer_norm" not in merge_k:
                    input_rep_k = merge_k[:-7]
                    
                    gram = input_representations[input_rep_k]

                    gram = gram + rescale_gram_matrice(gram, 0)
                    
                    gram = rescale_gram_matrice(gram, 0)

                    total_gram = total_gram + gram

                    # print("regmean: ", merge_k)

                    weight_to_merge = weight_to_merge + torch.matmul(weights_to_distill[merge_k], gram)
                else: # simple average for bias and ln
                    # print("direct merge: ", merge_k)
                    weight_to_merge = weight_to_merge + weights_to_distill[merge_k] / len(block_ids_to_distill)

        if not isinstance(total_gram, int): # weight
            # torch.linalg.pinv
            # print(k)
            # normalization_factor = torch.linalg.pinv(total_gram, hermitian=True)
            # weight_to_merge = torch.matmul(weight_to_merge, normalization_factor)
            weight_to_merge = weight_to_merge + scaling_factor * output_representations[input_rep_k]

            weight_to_merge = torch.linalg.lstsq(total_gram.T, weight_to_merge.T).solution.T

        # if not torch.all(torch.isclose(weight_to_merge, weights_to_distill[merge_k])):
        #     print(merge_k, (weight_to_merge - weights_to_distill[merge_k]).abs().mean())

        weights[k] = weight_to_merge

    return weights


def weights_multiplication_from_different_layers(weights_to_distill, target_names, distilled_block_ids):
    weights = {}

    for k in target_names:
        current_block = re.findall("block[.][0-9]*", k)
        assert len(current_block) <= 1
        current_block = current_block[0] if len(current_block) > 0 else None

        if current_block is None or "relative_attention_bias" in k:
            weights[k] = weights_to_distill[k]
            continue

        current_block_id = int(current_block.split(".")[-1])

        block_ids_to_distill = distilled_block_ids[current_block_id]

        if isinstance(block_ids_to_distill, int):
            block_ids_to_distill = [block_ids_to_distill]

        block_ids_to_distill = list(block_ids_to_distill)

        weight_to_merge = 1

        print(block_ids_to_distill)

        for block_id in block_ids_to_distill:
            merge_block = f"block.{block_id}"

            merge_k = k.replace(current_block, merge_block)

            weight_to_merge = weight_to_merge * weights_to_distill[merge_k]

        weights[k] = weight_to_merge

    return weights


def t5_modify_with_weight_init(transformer, petl_config, derivative_info=None, sampled_loader=None, pruned_indices=None):

    device = list(transformer.parameters())[0].device

    transformer.to("cpu")

    t5_prune_indices = None

    if petl_config.side_pretrained_weight is not None:
        is_strct_pruning = "unstrct" in petl_config.distillation_init

        side_config = deepcopy(transformer.config)

        num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = petl_config.side_pretrained_weight.split("-")

        num_layers = int(num_layers)
        res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = float(res_keep_ratio), float(attn_keep_ratio), float(ffn_keep_ratio)

        side_config.num_decoder_layers = num_layers
        side_config.num_layers = num_layers

        if is_strct_pruning:
            # unstructural
            side_config.d_model = side_config.d_model
            side_config.d_ff = side_config.d_ff
            side_config.d_kv = side_config.d_kv
        else:
            # structural
            side_config.d_model = get_pruned_dim(side_config.d_model, res_keep_ratio)
            side_config.d_ff = get_pruned_dim(side_config.d_ff, ffn_keep_ratio)
            side_config.d_kv = get_pruned_dim(side_config.d_kv, attn_keep_ratio)

        distilled_transformer = transformer.__class__(side_config)

        layers_to_extract_per_layer = transformer.config.num_layers // side_config.num_layers

        print(transformer.config.num_layers, side_config.num_layers)

        if petl_config.permute_before_merge:
            print("Start permutation (based on the first layer)...")
            transformer = permute_based_on_first_layer(transformer)

        elif petl_config.permute_on_block_before_merge:
            print("Start permutation (based on block)...")
            transformer = permute_based_on_block(transformer, eval(petl_config.distilled_block_ids))

        weights = None

        if petl_config.distillation_init:
            if "sum" in petl_config.distillation_init:

                weights = weights_interpolation_from_different_layers(
                    transformer.state_dict(), 
                    distilled_transformer.state_dict().keys(),
                    eval(petl_config.distilled_block_ids),
                    eval(petl_config.distilled_block_weights) if petl_config.distilled_block_weights is not None else None,
                    petl_config.modules_to_merge,
                )
            elif "mul" in petl_config.distillation_init:
                weights = weights_multiplication_from_different_layers(
                    transformer.state_dict(), 
                    distilled_transformer.state_dict().keys(),
                    eval(petl_config.distilled_block_ids),
                )

            elif "regmean-distill" in petl_config.distillation_init:
                cache_type = "input_output_gram"
                representations = get_activations(transformer, sampled_loader, eval(petl_config.distilled_block_ids), cache_type=cache_type)
                weights = weights_regmean_distill_from_different_layers(
                    transformer.state_dict(), 
                    distilled_transformer.state_dict().keys(),
                    eval(petl_config.distilled_block_ids),
                    representations,
                    petl_config.scaling_factor
                )

            elif "regmean" in petl_config.distillation_init:
                cache_type = "input_gram"
                representations = get_activations(transformer, sampled_loader, eval(petl_config.distilled_block_ids), cache_type=cache_type)
                weights = weights_regmean_from_different_layers(
                    transformer.state_dict(), 
                    distilled_transformer.state_dict().keys(),
                    eval(petl_config.distilled_block_ids),
                    representations,
                    petl_config.scaling_factor
                )

            if weights is not None:
                distilled_transformer.load_state_dict(weights)

            if "rep" in petl_config.distillation_init:

                layer_ids_list = get_layer_ids(eval(petl_config.distilled_block_ids))
                print("rep_stack_forward:", layer_ids_list)
                distilled_transformer.encoder.forward = functools.partial(rep_stack_forward, distilled_transformer.encoder)
                distilled_transformer.decoder.forward = functools.partial(rep_stack_forward, distilled_transformer.decoder)

                distilled_transformer.encoder.layer_ids_list = layer_ids_list
                distilled_transformer.decoder.layer_ids_list = layer_ids_list

            if "learnable" in petl_config.distillation_init:
                distilled_transformer = t5_modify_for_learnable_merge(
                    transformer, 
                    distilled_transformer, 
                    eval(petl_config.distilled_block_ids), 
                    petl_config.learnable_weight_type
                )

            if pruned_indices is not None:
                print("Use pre-extracted pruned indices...")
                if is_strct_pruning:
                    distilled_transformer, t5_prune_indices = t5_unstrct_pruning(
                        transformer, 
                        distilled_transformer, 
                        None,
                        res_keep_ratio, 
                        attn_keep_ratio, 
                        ffn_keep_ratio,
                        is_global="global" in petl_config.distillation_init,
                        pruned_indices=pruned_indices,
                    )
                else:    
                    distilled_transformer, t5_prune_indices = pruning(
                        transformer, 
                        distilled_transformer, 
                        None,
                        res_keep_ratio, 
                        attn_keep_ratio, 
                        ffn_keep_ratio,
                        is_global="global" in petl_config.distillation_init,
                        pruned_indices=pruned_indices,
                    )

            elif "prune" in petl_config.distillation_init:
                if derivative_info is not None:
                    # add back some weight which is shared-weight
                    for n, p in transformer.state_dict().items():
                        if n not in derivative_info: # those are shared embeddings
                            print(f"doesn't have {n}. Use shared.weight for it.")
                            derivative_info[n] = derivative_info["shared.weight"]

                if "mag_prune" in petl_config.distillation_init:
                    # using square of magnitude as the measure.
                    importance_measure = {k: v ** 2 for k, v in transformer.state_dict().items()}

                elif "derv_prune" in petl_config.distillation_init:
                    importance_measure = derivative_info

                elif "obs_prune" in petl_config.distillation_init:
                    importance_measure = {k: (v ** 2) * derivative_info[k] for k, v in transformer.state_dict().items()}

                elif "zero_prune" in petl_config.distillation_init:
                    importance_measure = {k: torch.zeros_like(v) for k, v in transformer.state_dict().items()}

                elif "rand_prune" in petl_config.distillation_init:
                    # all the importance is random
                    importance_measure = {k: torch.randn_like(v) for k, v in transformer.state_dict().items()}

                else:
                    raise ValueError("The pruning method is invalid.")

                print(f"Apply {petl_config.distillation_init}...")

                if is_strct_pruning:
                    distilled_transformer, t5_prune_indices = t5_unstrct_pruning(
                        transformer, 
                        distilled_transformer, 
                        importance_measure,
                        res_keep_ratio, 
                        attn_keep_ratio, 
                        ffn_keep_ratio,
                        is_global="global" in petl_config.distillation_init,
                    )
                else:
                    distilled_transformer, t5_prune_indices = pruning(
                        transformer, 
                        distilled_transformer, 
                        importance_measure,
                        res_keep_ratio, 
                        attn_keep_ratio, 
                        ffn_keep_ratio,
                        is_global="global" in petl_config.distillation_init,
                    )

            if "fusion" in petl_config.distillation_init:
                print("Apply fusion...")
                distilled_transformer = fusion(
                    transformer, 
                    distilled_transformer, 
                    distill_merge_ratio=petl_config.distill_merge_ratio, 
                    exact=petl_config.exact, 
                    normalization=petl_config.normalization, 
                    metric=petl_config.metric,
                    to_one=petl_config.to_one,
                    importance=petl_config.importance,
                )

        distilled_transformer.to(device)

        del transformer
        return distilled_transformer, t5_prune_indices
    
    transformer.to(device)
    return transformer, t5_prune_indices


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

            self.side_pretrained_weight = "6-0.8-1.0-1.0"
            self.distillation_init = "unstrct_mag_prune"
            self.distilled_block_ids = "[[0,1,2,3],[2,3,4,5],[4,5,6,7],[6,7,8,9],[8,9,10,11],[10,11]]" # "[0,1,2,3,4,[5,6,7,8,9,10,11]]" # "[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]" "[0,1,2,[3,4,5],[6,7,8],[9,10,11]]"
            self.distilled_block_weights = None
            self.rep_stack_forward = True
            self.scaling_factor = 1.0
            self.learnable_weight_type = "scalar-shared"
            self.modules_to_merge = ".*|.*" # ".*layer_norm.*|.*DenseReluDense.*" # ".*|.*" 
            self.permute_before_merge = False
            self.permute_on_block_before_merge = False
            self.distill_merge_ratio = 0.5
            self.exact = True
            self.normalization = False
            self.metric = "dot"
            self.to_one = False
            self.importance = True

    config = AdapterConfig(args.adapter_type)
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small") # google/flan-t5-xl
    tokenizer = AutoTokenizer.from_pretrained("t5-small") # google/flan-t5-xl

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

    generation_input_ids = tokenizer(["translate English to German: The house is wonderful, and the garden is really big too. Therefore, I like the house in general."], return_tensors="pt")
    old_generation_outputs = model.generate(**generation_input_ids, num_beams=1)


    if "regmean" in config.distillation_init:

        class MyDataset(torch.utils.data.Dataset):
            def __init__(self, inputs, labels):
                self.inputs = inputs
                self.labels = labels
            
            def __len__(self):
                return self.inputs.shape[0]

            def __getitem__(self, idx):
                return {'input_ids': self.inputs[idx], "labels": self.labels[idx]}

        sampled_loader = torch.utils.data.DataLoader(
            MyDataset(torch.randint(1, 1000, (128, 35)), torch.randint(1, 1000, (128, 5))),
            batch_size=32
        )
    else:
        sampled_loader = None
    
    derivative_info = {k: v ** 2 for k, v in model.state_dict().items()}

    model, _ = t5_modify_with_weight_init(model, config, derivative_info, sampled_loader)
    new_param = model.state_dict()
    # for i in new_param.keys():
    #     print(i)
        # if "adapter" in i:
        #     print(i, new_param[i])

    print("New model")
    # print(model)
    model.eval()
    with torch.no_grad():
        new_outputs = model(
            input_ids=input_seq.input_ids,
            decoder_input_ids=target_seq.input_ids[:, :-1],
            labels=target_seq.input_ids[:, 1:],
        )

    print("Trainable parameters")
    print(
        [
            p_name
            for p_name in dict(model.named_parameters()).keys()
            if re.fullmatch(config.trainable_param_names, p_name)
        ]
    )
    print(f"Logits diff {torch.abs(old_outputs.logits - new_outputs.logits).mean():.3f}")
    print(f"Loss diff old={old_outputs.loss:.3f} new={new_outputs.loss:.3f}")

    print(generation_input_ids.input_ids.shape)

    new_generation_outputs = model.generate(**generation_input_ids, num_beams=1)

    print(old_generation_outputs)
    print(new_generation_outputs)

    print(tokenizer.batch_decode(old_generation_outputs))
    print(tokenizer.batch_decode(new_generation_outputs))

    # print("Before converting")
    # for i in model.state_dict().keys():
    #     print(i)

    # convert_to_normal_save_weights(model)

    # print("After converting")
    # for i in model.state_dict().keys():
    #     print(i)
