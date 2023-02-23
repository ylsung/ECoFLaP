import torch
import torch.nn as nn
from seq2seq.methods.adapters.adapter_modeling import Adapter


class T5LayerFFWithAdapter(nn.Module):
    def __init__(self, T5LayerFF, adapter_config, transformer_config):
        super().__init__()
        self.DenseReluDense = T5LayerFF.DenseReluDense
        self.adapter = Adapter(adapter_config, transformer_config)
        self.layer_norm = T5LayerFF.layer_norm
        self.dropout = T5LayerFF.dropout

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        adapter_output = self.adapter(forwarded_states)
        hidden_states = hidden_states + self.dropout(adapter_output)
        return hidden_states


class T5LayerSelfAttentionWithAdapter(nn.Module):
    def __init__(self, T5LayerSelfAttention, adapter_config, transformer_config):
        super().__init__()
        self.SelfAttention = T5LayerSelfAttention.SelfAttention
        self.adapter = Adapter(adapter_config, transformer_config)
        self.layer_norm = T5LayerSelfAttention.layer_norm
        self.dropout = T5LayerSelfAttention.dropout

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        adapter_output = self.adapter(attention_output[0])
        hidden_states = hidden_states + self.dropout(adapter_output)
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttentionWithAdapter(nn.Module):
    def __init__(self, T5LayerCrossAttention, adapter_config, transformer_config):
        super().__init__()
        self.EncDecAttention = T5LayerCrossAttention.EncDecAttention
        self.adapter = Adapter(adapter_config, transformer_config)
        self.layer_norm = T5LayerCrossAttention.layer_norm
        self.dropout = T5LayerCrossAttention.dropout

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        adapter_output = self.adapter(attention_output[0])
        layer_output = hidden_states + self.dropout(adapter_output)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs