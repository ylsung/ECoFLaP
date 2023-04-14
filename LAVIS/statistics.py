from transformers import T5ForConditionalGeneration
import re

from lavis.compression.weight_matching import permute_based_on_block

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from collections import defaultdict

import torch
import torch.nn as nn

import numpy as np


def return_unpair_cosine_dist(embed):
    dot = (embed.unsqueeze(1) * embed.unsqueeze(0)).sum(-1)

    l_length = (embed ** 2).sum(-1) ** 0.5
    v_length = (embed ** 2).sum(-1) ** 0.5

    length_prod = l_length.unsqueeze(1) * v_length.unsqueeze(0)

    cosine_similarity = dot / length_prod

    mask = 1 - torch.eye(embed.shape[0])

    return cosine_similarity * mask


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

generation_input_ids = tokenizer(["translate English to German: The house is wonderful, and the garden is really big too. Therefore, I like the house in general."], return_tensors="pt")
old_generation_outputs = model.generate(**generation_input_ids, num_beams=1)


model = permute_based_on_block(model, range(24))

new_generation_outputs = model.generate(**generation_input_ids, num_beams=1)

print(old_generation_outputs)
print(new_generation_outputs)

print(tokenizer.batch_decode(old_generation_outputs))
print(tokenizer.batch_decode(new_generation_outputs))

state_dict = model.state_dict()


encoder_keys = {
    "encoder.block.{}.layer.0.SelfAttention.q.weight",                                                                                                
    "encoder.block.{}.layer.0.SelfAttention.k.weight",                                                                                                
    "encoder.block.{}.layer.0.SelfAttention.v.weight",                                                                                                
    "encoder.block.{}.layer.0.SelfAttention.o.weight",                                                                                                
    "encoder.block.{}.layer.1.DenseReluDense.wi_0.weight",
    "encoder.block.{}.layer.1.DenseReluDense.wi_1.weight",
    "encoder.block.{}.layer.1.DenseReluDense.wo.weight",
}

decoder_keys = {
    "decoder.block.{}.layer.0.SelfAttention.q.weight",                                                                                                
    "decoder.block.{}.layer.0.SelfAttention.k.weight",                                                                                                
    "decoder.block.{}.layer.0.SelfAttention.v.weight",                                                                                                
    "decoder.block.{}.layer.0.SelfAttention.o.weight",                                                                                                
    "decoder.block.{}.layer.1.EncDecAttention.q.weight",                                                                                              
    "decoder.block.{}.layer.1.EncDecAttention.k.weight",                                                                                              
    "decoder.block.{}.layer.1.EncDecAttention.v.weight",
    "decoder.block.{}.layer.1.EncDecAttention.o.weight",
    "decoder.block.{}.layer.2.DenseReluDense.wi_0.weight",
    "decoder.block.{}.layer.2.DenseReluDense.wi_1.weight",
    "decoder.block.{}.layer.2.DenseReluDense.wo.weight",
}

cos = nn.CosineSimilarity(dim=1)

sim_dict = defaultdict(lambda: defaultdict(list))


for sub_model in ["encoder", "decoder"]:
    if sub_model == "encoder":
        module_keys = encoder_keys
    else:
        module_keys = decoder_keys

    for layer_id in range(model.config.num_layers):
        for other_layer_id in range(model.config.num_layers):
            for module_key in module_keys:
                current_module_key, other_module_key = module_key.format(layer_id), module_key.format(other_layer_id)
                current_module_weight, other_module_weight = state_dict[current_module_key], state_dict[other_module_key]

                cos_sim = cos(current_module_weight, other_module_weight).mean()

                sim_dict[f"{sub_model}.block.{layer_id}"][current_module_key].append(cos_sim)


block_sim = {"encoder": [], "decoder": []}

for sub_model in ["encoder", "decoder"]:
    for layer_id in range(model.config.num_layers):
        block_sim[sub_model].append([])
        for other_layer_id in range(model.config.num_layers):
            
            sim_avg = np.mean([sim_dict[f"{sub_model}.block.{layer_id}"][k][other_layer_id].item() for k in sim_dict[f"{sub_model}.block.{layer_id}"].keys()])

            block_sim[sub_model][-1].append(sim_avg)

print(block_sim)


# sim_dict = defaultdict(lambda: defaultdict(float))


# for sub_model in ["encoder", "decoder"]:
#     if sub_model == "encoder":
#         module_keys = encoder_keys
#     else:
#         module_keys = decoder_keys

#     for layer_id in range(model.config.num_layers):
#         for module_key in module_keys:
#             current_module_key = module_key.format(layer_id)
#             current_module_weight = state_dict[current_module_key]

#             valid_num = current_module_weight.shape[0] ** 2 - current_module_weight.shape[0]

#             cos_sim = return_unpair_cosine_dist(current_module_weight).abs().sum() / valid_num

#             sim_dict[f"{sub_model}.block.{layer_id}"][current_module_key] = cos_sim

#             print(current_module_key, cos_sim)
