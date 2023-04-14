from transformers import T5ForConditionalGeneration
import re

from lavis.compression.weight_matching import permute_based_on_block

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from collections import defaultdict

import torch
import torch.nn as nn

import numpy as np

import torch_pruning as tp

import torchvision


model = torchvision.models.vit_b_16()

# model = permute_based_on_block(model, range(24))

# Importance criteria
imp = tp.importance.MagnitudeImportance(p=2)

ignored_layers = []

for name, m in model.named_modules():
    if isinstance(m, torch.nn.Linear) and "head" in name:
        print(name)
        ignored_layers.append(m) # DO NOT prune the final classifier!


example_inputs = torch.randn(1, 3, 224, 224)

print(model(example_inputs))

iterative_steps = 5 # progressive pruning
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    iterative_steps=iterative_steps,
    ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    round_to=12,
    ignored_layers=ignored_layers,
)

# base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
print(model)
for i in range(iterative_steps):
    pruner.step()

print(model)
    # macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)

print(model(example_inputs))
