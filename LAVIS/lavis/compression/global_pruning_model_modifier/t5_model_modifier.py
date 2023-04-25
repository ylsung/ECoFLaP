# codes adapted from https://github.com/r-three/t-few/blob/4e581fa0b8f53e36da252a15bd581d365d4dd333/src/models/adapters.py
import torch
import torch.nn as nn
import re
from copy import deepcopy

from lavis.models.blip2_models.modeling_t5 import T5LayerFF, T5LayerSelfAttention, T5LayerCrossAttention


def t5_modify_global_pruning(transformer, pruned_state_dict):
    temp_config = deepcopy(transformer.config)
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(".*block[.][0-9]*", m_name):
            # layer = nn.ModuleList()

            # temp_config.d_kv = pruned_state_dict[m_name + ".layer.0.SelfAttention.q.weight"].shape[0] // temp_config.num_heads

            # layer.append(
            #     T5LayerSelfAttention(temp_config, , has_relative_attention_bias=getattr(module.layer[0].SelfAttention, "relative_attention_bias", None) is not None)
            # )

            d_model = pruned_state_dict[m_name + ".layer.0.SelfAttention.q.weight"].shape[1]
            inner_dim = pruned_state_dict[m_name + ".layer.0.SelfAttention.q.weight"].shape[0]

            module.layer[0].SelfAttention.q = nn.Linear(d_model, inner_dim, bias=False)
            module.layer[0].SelfAttention.k = nn.Linear(d_model, inner_dim, bias=False)
            module.layer[0].SelfAttention.v = nn.Linear(d_model, inner_dim, bias=False)
            module.layer[0].SelfAttention.o = nn.Linear(inner_dim, d_model, bias=False)

            # if getattr(module.layer[0].SelfAttention, "relative_attention_bias", None) is not None:
            #     layer[0].SelfAttention.relative_attention_bias = module.layer[0].SelfAttention.relative_attention_bias

            if module.is_decoder:

                # temp_config.d_kv = pruned_state_dict[m_name + ".layer.1.EncDecAttention.q.weight"].shape[0] // temp_config.num_heads
                # layer.append(
                #     T5LayerCrossAttention(temp_config)
                # )

                # layer.append(module.layer[1])

                d_model = pruned_state_dict[m_name + ".layer.1.EncDecAttention.q.weight"].shape[1]
                inner_dim = pruned_state_dict[m_name + ".layer.1.EncDecAttention.q.weight"].shape[0]

                module.layer[1].EncDecAttention.q = nn.Linear(d_model, inner_dim, bias=False)
                module.layer[1].EncDecAttention.k = nn.Linear(d_model, inner_dim, bias=False)
                module.layer[1].EncDecAttention.v = nn.Linear(d_model, inner_dim, bias=False)
                module.layer[1].EncDecAttention.o = nn.Linear(inner_dim, d_model, bias=False)

            if module.is_decoder:
                # temp_config.d_ff = pruned_state_dict[m_name + ".layer.2.DenseReluDense.wo.weight"].shape[1]

                d_model = pruned_state_dict[m_name + ".layer.2.DenseReluDense.wo.weight"].shape[0]
                d_ff = pruned_state_dict[m_name + ".layer.2.DenseReluDense.wo.weight"].shape[1]
            else:
                # temp_config.d_ff = pruned_state_dict[m_name + ".layer.1.DenseReluDense.wo.weight"].shape[1]
                d_model = pruned_state_dict[m_name + ".layer.1.DenseReluDense.wo.weight"].shape[0]
                d_ff = pruned_state_dict[m_name + ".layer.1.DenseReluDense.wo.weight"].shape[1]

            if getattr(module.layer[-1].DenseReluDense, "wi_0", None) is not None:
                module.layer[-1].DenseReluDense.wi_0 = nn.Linear(d_model, d_ff, bias=False)
                module.layer[-1].DenseReluDense.wi_1 = nn.Linear(d_model, d_ff, bias=False)
                module.layer[-1].DenseReluDense.wo = nn.Linear(d_ff, d_model, bias=False)

            else:
                module.layer[-1].DenseReluDense.wi = nn.Linear(d_model, d_ff, bias=False)
                module.layer[-1].DenseReluDense.wo = nn.Linear(d_ff, d_model, bias=False)

            # layer.append(
            #     T5LayerFF(temp_config)
            # )
            # module.layer = layer

    return transformer