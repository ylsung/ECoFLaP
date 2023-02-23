"""Implements an Adapter, Low-rank adapters and Hyper-adapter Layers."""
# codes adapted from https://github.com/r-three/t-few/blob/master/src/models/AdapterVariants/Adapters.py
import torch.nn as nn
from transformers.activations import ACT2FN


ACT2FN["identity"] = lambda x: x


class Adapter(nn.Module):
    def __init__(self, adapter_config, transformer_config):
        super().__init__()
        self.adapter_input_size = transformer_config.hidden_size
        self.adapter_latent_size = self.adapter_input_size // adapter_config.adapter_reduction_factor
        self.non_linearity = ACT2FN[adapter_config.adapter_non_linearity]

        # down projection
        self.down_proj = nn.Linear(self.adapter_input_size, self.adapter_latent_size)
        # up projection
        self.up_proj = nn.Linear(self.adapter_latent_size, self.adapter_input_size)

        self.init_weights()

    def init_weights(self):
        """Initialize the weights -> so that initially we the whole Adapter layer is a near-identity function"""
        self.down_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.down_proj.bias.data.zero_()
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.bias.data.zero_()

    def forward(self, x):
        output = self.up_proj(self.non_linearity(self.down_proj(x)))
        output = x + output
        return output
