# codes adapted from https://github.com/r-three/t-few/blob/4e581fa0b8f53e36da252a15bd581d365d4dd333/src/models/adapters.py
import torch
import torch.nn as nn
import re

from seq2seq.methods.adapters.t5_module_with_adapter import T5LayerSelfAttentionWithAdapter, T5LayerCrossAttentionWithAdapter, T5LayerFFWithAdapter


def t5_modify_with_adapters(transformer, petl_config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(".*block[.][0-9]*", m_name):
            layer = nn.ModuleList()
            layer.append(
                T5LayerSelfAttentionWithAdapter(
                    module.layer[0],
                    petl_config,
                    transformer.config,
                )
            )
            if module.is_decoder:
                layer.append(
                    T5LayerCrossAttentionWithAdapter(
                        module.layer[1],
                        petl_config,
                        transformer.config,
                    )
                )
            layer.append(
                T5LayerFFWithAdapter(
                    module.layer[2] if module.is_decoder else module.layer[1],
                    petl_config,
                    transformer.config,
                )
            )
            module.layer = layer

    return transformer


if __name__ == "__main__":
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--adapter_type",
        required=True,
        type=str,
        choices=["normal", "lowrank", "compacter"],
    )
    args = parser.parse_args()

    class AdapterConfig:
        def __init__(self, adapter_type):
            self.adapter_type = adapter_type

            # Adapter Config
            self.adapter_reduction_factor = 16
            self.adapter_non_linearity = "relu"
            self.normal_adapter_residual = True
            self.add_compacter_in_attention = True

            self.trainable_param_names = ".*layer_norm.*|.*adapter.*"

    config = AdapterConfig(args.adapter_type)
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
    # print(model)
    # print(model.state_dict().keys())
    old_param = model.state_dict()
    with torch.no_grad():
        old_outputs = model(
            input_ids=input_seq.input_ids,
            decoder_input_ids=target_seq.input_ids[:, :-1],
            labels=target_seq.input_ids[:, 1:],
        )

    model = t5_modify_with_adapters(model, config)
    new_param = model.state_dict()
    # for i in new_param.keys():
    #     if "adapter" in i:
    #         print(i, new_param[i])
    
    for p_name in dict(model.named_parameters()).keys():
        if re.fullmatch(".*adapter.*", p_name):
            new_param[p_name].zero_()

    for i in new_param.keys():
        if "adapter" in i:
            print(i, new_param[i])

    print("New model")
    # print(model)
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

