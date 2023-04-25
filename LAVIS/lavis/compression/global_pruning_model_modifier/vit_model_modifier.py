# codes adapted from https://github.com/r-three/t-few/blob/4e581fa0b8f53e36da252a15bd581d365d4dd333/src/models/adapters.py
import torch
import torch.nn as nn
import re
from copy import deepcopy

from lavis.models.eva_vit import Mlp


def vit_modify_global_pruning(transformer, pruned_state_dict):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(".*blocks[.][0-9]*", m_name):

            print(m_name)

            in_features = pruned_state_dict[m_name + ".mlp.fc1.weight"].shape[1]
            hidden_features = pruned_state_dict[m_name + ".mlp.fc1.weight"].shape[0]
            out_features = pruned_state_dict[m_name + ".mlp.fc2.weight"].shape[0]

            module.mlp.fc1 = nn.Linear(in_features, hidden_features)
            module.mlp.fc2 = nn.Linear(hidden_features, out_features)

            print(hidden_features)

    return transformer