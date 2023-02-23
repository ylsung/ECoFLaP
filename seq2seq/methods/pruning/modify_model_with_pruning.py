import torch
import torch.nn as nn

import re
from copy import deepcopy
import functools

from seq2seq.methods.pbp.t5_module_with_pbp import (
    T5BlockPrefixWrapper,
    T5StackPrefixWrapper,
    T5LayerSelfAttentionPrefixWrapper,
    T5LayerCrossAttentionPrefixWrapper,
    T5AttentionPrefixWrapper,
    T5StackPBPWrapper, 
    T5ForConditionalGenerationPBPWrapper, 
    GenerationMixinPBPWrapper,
)

from transformers.models.t5.modeling_t5 import T5LayerSelfAttention, T5LayerCrossAttention

from seq2seq.methods.pruning.pruning_methods import pruning_v2


def t5_modify_with_pbp(transformer, petl_config):

    # Get the pruned weights for initializing the side network
    pruned_state_dict = pruning_v2(transformer, petl_config.pbp_reduction_factor)

    side_config = deepcopy(transformer.config)

    side_config.d_model //= petl_config.pbp_reduction_factor
    side_config.d_ff //= petl_config.pbp_reduction_factor
    side_config.d_kv //= petl_config.pbp_reduction_factor

    side_transformer = transformer.__class__(side_config)

    backbone_modules = dict(transformer.named_modules())
    side_modules = dict(side_transformer.named_modules())


    # overwrite the forward function of encoders and decoders for introducing prefix 
    transformer.encoder.forward = functools.partial(T5StackPrefixWrapper.forward, transformer.encoder)
    transformer.decoder.forward = functools.partial(T5StackPrefixWrapper.forward, transformer.decoder)

    side_transformer.encoder.forward = functools.partial(T5StackPrefixWrapper.forward, side_transformer.encoder)
    side_transformer.decoder.forward = functools.partial(T5StackPrefixWrapper.forward, side_transformer.decoder)

    transformer.encoder = T5StackPBPWrapper(transformer.encoder, side_transformer.encoder, transformer.config, side_config, petl_config)
    transformer.decoder = T5StackPBPWrapper(transformer.decoder, side_transformer.decoder, transformer.config, side_config, petl_config)

    transformer.upsample = nn.Linear(side_config.d_model, transformer.config.d_model) # add the upsample layer
    transformer.forward = functools.partial(T5ForConditionalGenerationPBPWrapper.forward, transformer) # assign the self to the module (T5ForConditionalGeneration) object
    transformer._expand_inputs_for_generation = GenerationMixinPBPWrapper._expand_inputs_for_generation # for enable beam search for LST
    # transformer._reorder_cache = functools.partial(T5ForConditionalGenerationPBPWrapper._reorder_cache, transformer) # for enable beam search for LST

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(".*block[.][0-9]*", m_name):
            module.forward = functools.partial(T5BlockPrefixWrapper.forward, module) # assign the self to the module (T5Block) object

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(".*SelfAttention|.*EncDecAttention", m_name):
            module.forward = functools.partial(T5AttentionPrefixWrapper.forward, module) # assign the self to the module (T5Attention) object

    for m_name, module in dict(transformer.named_modules()).items():
        if isinstance(module, T5LayerSelfAttention):
            module.forward = functools.partial(T5LayerSelfAttentionPrefixWrapper.forward, module) # assign the self to the module (T5LayerSelfAttention) object

        elif isinstance(module, T5LayerCrossAttention):
            module.forward = functools.partial(T5LayerCrossAttentionPrefixWrapper.forward, module) # assign the self to the module (T5LayerCrossAttention) object

    for p_name, param in transformer.named_parameters():
        if re.fullmatch(".*side.*", p_name) and "embed_tokens" not in p_name:
            corresponding_p_name_in_pruned_params_dict = p_name.replace(".side.", ".")

            pruned_weight = pruned_state_dict[corresponding_p_name_in_pruned_params_dict]
            param.data.copy_(pruned_weight)

    return transformer


if __name__ == "__main__":
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import argparse

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
            self.pbp_reduction_factor = 16
            self.trainable_param_names = ".*prompts.*|.*embed_to_kv.*|.*upsample.*|.*side.*"

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
    old_generation_outputs = model.generate(**generation_input_ids, num_beams=5)

    model = t5_modify_with_pbp(model, config)
    new_param = model.state_dict()
    # for i in new_param.keys():
    #     if "adapter" in i:
    #         print(i, new_param[i])

    # for p_name, param in model.named_parameters():

    #     if re.fullmatch(".*downsample.*|.*upsample.*", p_name):
    #         if p_name.endswith("weight"):
    #             param.data = torch.eye(new_param[p_name].shape[0])
    #         elif p_name.endswith("bias"):
    #             param.data.zero_()
    #         else:
    #             raise NotImplementedError

    # for p_name, param in model.named_parameters():
    #     if re.fullmatch(config.trainable_param_names, p_name):
    #         print(p_name)

    print("New model")
    # print(model)
    model.eval()
    with torch.no_grad():
        new_outputs = model(
            input_ids=input_seq.input_ids,
            decoder_input_ids=target_seq.input_ids[:, :-1],
            labels=target_seq.input_ids[:, 1:],
        )

    # print("Trainable parameters")
    # print(
    #     [
    #         p_name
    #         for p_name in dict(model.named_parameters()).keys()
    #         if re.fullmatch(config.trainable_param_names, p_name)
    #     ]
    # )
    print(f"Logits diff {torch.abs(old_outputs.logits - new_outputs.logits).mean():.3f}")
    print(f"Loss diff old={old_outputs.loss:.3f} new={new_outputs.loss:.3f}")

    print(generation_input_ids.input_ids.shape)

    new_generation_outputs = model.generate(**generation_input_ids, num_beams=1)

    print(old_generation_outputs)
    print(new_generation_outputs)
