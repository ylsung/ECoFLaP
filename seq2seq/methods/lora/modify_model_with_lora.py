# Codes borrowed from https://github.com/r-three/t-few/blob/master/src/models/lora.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from seq2seq.methods.lora.module_with_lora import LoRALinear


def modify_with_lora(transformer, petl_config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(petl_config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(petl_config.lora_layers, c_name):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        LoRALinear(layer, petl_config.lora_rank, petl_config.lora_scaling_rank, petl_config.lora_init_scale),
                    )
    return transformer


if __name__ == "__main__":
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    class LoRAConfig:
        def __init__(self):
            self.lora_rank = 4
            self.lora_init_scale = 0.01
            self.lora_modules = ".*SelfAttention|.*EncDecAttention"
            self.lora_layers = "q|k|v|o"
            self.trainable_param_names = ".*layer_norm.*|.*lora_[ab].*"
            self.lora_scaling_rank = 1
            # lora_modules and lora_layers are speicified with regular expressions
            # see https://www.w3schools.com/python/python_regex.asp for reference

    config = LoRAConfig()
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    input_seq = tokenizer(
        ["Applies a linear transformation to the incoming data."],
        return_tensors="pt",
    )
    target_seq = tokenizer(
        ["Parameters: in_features - size of each input sample. out_features - size of each output sample."],
        return_tensors="pt",
    )

    print("Old model")
    print(model)
    with torch.no_grad():
        old_outputs = model(
            input_ids=input_seq.input_ids,
            decoder_input_ids=target_seq.input_ids[:, :-1],
            labels=target_seq.input_ids[:, 1:],
        )

    model = modify_with_lora(model, config)

    print("New model")
    print(model)
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