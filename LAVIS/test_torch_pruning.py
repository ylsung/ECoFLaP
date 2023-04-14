from transformers import T5ForConditionalGeneration
import re

from lavis.compression.weight_matching import permute_based_on_block

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from collections import defaultdict

import torch
import torch.nn as nn

import numpy as np

import torch_pruning as tp


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

generation_input_ids = tokenizer(["translate English to German: The house is wonderful, and the garden is really big too. Therefore, I like the house in general."], return_tensors="pt")
old_generation_outputs = model.generate(**generation_input_ids, num_beams=1)


# model = permute_based_on_block(model, range(24))

# Importance criteria
imp = tp.importance.MagnitudeImportance(p=2)

ignored_layers = []

for name, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and "lm_head" in name:
        print(name)
        ignored_layers.append(m) # DO NOT prune the final classifier!

input_seq = tokenizer(
    ["Applies a linear transformation to the incoming data."],
    return_tensors="pt",
)
target_seq = tokenizer(
    ["Parameters: in_features - size of each input sample. out_features - size of each output sample."],
    return_tensors="pt",
)

example_inputs = {
    "input_ids": input_seq.input_ids,
    "decoder_input_ids": target_seq.input_ids[:, :-1],
    "labels": target_seq.input_ids[:, 1:],
}

iterative_steps = 5 # progressive pruning
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    iterative_steps=iterative_steps,
    ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    round_to=model.config.num_heads,
    ignored_layers=ignored_layers,
)

# base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
print(model.encoder)

for i in range(iterative_steps):
    pruner.step()
    
    # macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)

print(model.encoder)

new_generation_outputs = model.generate(**generation_input_ids, num_beams=1)

print(old_generation_outputs)
print(new_generation_outputs)

print(tokenizer.batch_decode(old_generation_outputs))
print(tokenizer.batch_decode(new_generation_outputs))