import torch
import torch.nn as nn

import re
from copy import deepcopy
import functools

from seq2seq.methods.prefix_tuning.t5_module_with_prefix_tuning import (
    T5BlockPrefixWrapper,
    T5StackPrefixWrapper,
    T5LayerSelfAttentionPrefixWrapper,
    T5LayerCrossAttentionPrefixWrapper,
    T5AttentionPrefixWrapper,
    T5StackForPrefix, 
)

from transformers.models.t5.modeling_t5 import T5LayerSelfAttention, T5LayerCrossAttention

from transformers import AutoTokenizer


def t5_modify_with_prefix_tuning(transformer, petl_config):

    # Get tokenizer for processing prompts
    tokenizer = AutoTokenizer.from_pretrained(transformer.config._name_or_path)

    transformer.encoder.forward = functools.partial(T5StackPrefixWrapper.forward, transformer.encoder)
    transformer.decoder.forward = functools.partial(T5StackPrefixWrapper.forward, transformer.decoder)

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


    transformer.encoder = T5StackForPrefix(transformer.encoder, transformer.config, petl_config)
    transformer.decoder = T5StackForPrefix(transformer.decoder, transformer.config, petl_config)

    # set hard prompts
    if petl_config.hard_prompt is not None:
        hard_prompt_input_ids = tokenizer(
            [petl_config.hard_prompt],
            return_tensors="pt",
        ).input_ids
        transformer.encoder.set_hard_prompts(hard_prompt_input_ids)
        transformer.decoder.set_hard_prompts(hard_prompt_input_ids)

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
            self.trainable_param_names = ".*prompts.*|.*embed_to_kv.*|.*upsample.*|.*side.*|.*downsample.*"
            self.enc_num_tokens = 10
            self.dec_num_tokens = 10
            self.prompts_expand_after = True
            self.hard_prompt = None # "Find the relationship between the two concatenated sentences, or classify it if there is only one sentence."
            self.init_from_emb = True

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

    model = t5_modify_with_prefix_tuning(model, config)
    new_param = model.state_dict()
    # for i in new_param.keys():
    #     if "adapter" in i:
    #         print(i, new_param[i])


    ## print model's trainable parameters (for sanity check)
    # for p_name, param in model.named_parameters():
    #     if re.fullmatch(config.trainable_param_names, p_name):
    #         print(p_name)

    # for p_name, param in model.named_parameters():
    #     if "side" in p_name:
    #         orig_name = p_name.replace(".side.", ".")

    #         assert torch.all(param == old_param[orig_name])

    # # print model's trainable parameters (for sanity check)
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

    print(tokenizer.batch_decode(old_generation_outputs))
    print(tokenizer.batch_decode(new_generation_outputs))
