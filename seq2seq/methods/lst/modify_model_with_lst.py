import torch
import torch.nn as nn

import re
from copy import deepcopy
import functools

from seq2seq.methods.lst.t5_module_with_lst import (
    FunctionLSTWrapper, 
    T5BlockLSTWrapper, 
    T5StackLSTWrapper, 
    T5ForConditionalGenerationLSTWrapper, 
    GenerationMixinLSTWrapper,
)

from seq2seq.methods.pruning.pruning_methods import pruning_v2


def t5_modify_with_lst(transformer, petl_config):

    # Get the pruned weights for initializing the side network
    pruned_state_dict = pruning_v2(transformer, petl_config.lst_reduction_factor)

    side_config = deepcopy(transformer.config)

    side_config.d_model //= petl_config.lst_reduction_factor
    side_config.d_ff //= petl_config.lst_reduction_factor
    side_config.d_kv //= petl_config.lst_reduction_factor

    side_transformer = transformer.__class__(side_config)

    backbone_modules = dict(transformer.named_modules())
    side_modules = dict(side_transformer.named_modules())

    for m_name, module in dict(transformer.named_modules()).items():
        side_module = side_modules[m_name]
        if re.fullmatch("encoder|decoder", m_name):
            module.block = nn.ModuleList(
                [
                    T5BlockLSTWrapper(module.block[i], side_module.block[i], i==0, transformer.config, side_config, petl_config)
                    for i in range(len(module.block))
                ]
            )
            module.final_layer_norm = FunctionLSTWrapper(module.final_layer_norm, side_module.final_layer_norm)
            module.dropout = FunctionLSTWrapper(module.dropout, side_module.dropout)

            module.forward = functools.partial(T5StackLSTWrapper.forward, module) # assign the self to the module (T5Stack) object

    transformer.upsample = nn.Linear(side_config.d_model, transformer.config.d_model) # add the upsample layer
    transformer.forward = functools.partial(T5ForConditionalGenerationLSTWrapper.forward, transformer) # assign the self to the module (T5ForConditionalGeneration) object
    transformer._expand_inputs_for_generation = GenerationMixinLSTWrapper._expand_inputs_for_generation # for enable beam search for LST
    transformer._reorder_cache = functools.partial(T5ForConditionalGenerationLSTWrapper._reorder_cache, transformer) # for enable beam search for LST

    for p_name, param in transformer.named_parameters():
        if re.fullmatch(".*side.*", p_name) and "gate" not in p_name:
            print(p_name)
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
            self.lst_reduction_factor = 1
            self.trainable_param_names = ".*downsample.*|.*upsample.*|.*side.*"
            self.gate_T = 0.1
            self.gate_alpha = 0.0
            self.gate_type = "learnable"

    config = AdapterConfig(args.adapter_type)
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    input_seq = tokenizer(
        ["Applies a linear transformation to the incoming data."],
        return_tensors="pt",
    )
    target_seq = tokenizer(
        ["Parameters: in_features - size of each input sample. out_features - size of each output sample."],
        return_tensors="pt",
    )

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

    generation_input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt")
    old_generation_outputs = model.generate(**generation_input_ids, num_beams=5)

    model = t5_modify_with_lst(model, config)
    new_param = model.state_dict()
    # for i in new_param.keys():
    #     if "adapter" in i:
    #         print(i, new_param[i])

    for p_name, param in model.named_parameters():

        if re.fullmatch(".*downsample.*|.*upsample.*", p_name):
            if p_name.endswith("weight"):
                param.data = torch.eye(new_param[p_name].shape[0])
            elif p_name.endswith("bias"):
                param.data.zero_()
            else:
                raise NotImplementedError

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

    new_generation_outputs = model.generate(**generation_input_ids, num_beams=5)

    print(old_generation_outputs)
    print(new_generation_outputs)

