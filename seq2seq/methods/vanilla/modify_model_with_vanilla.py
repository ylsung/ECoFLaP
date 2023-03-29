import torch
import torch.nn as nn

import re
from copy import deepcopy
import functools

import transformers
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from typing import Optional, Tuple, Union, Dict, Any

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, BaseModelOutput

from transformers import AutoTokenizer


class T5AttentionVanilla(nn.Module):
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                attn_type = "self-attn 1"
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                attn_type = "cross-attn 1"
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                    attn_type = "self-attn 2"
                else:
                    # cross-attn
                    hidden_states = past_key_value
                    attn_type = "cross-attn 2"
            return hidden_states, attn_type

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states, attn_type = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states, attn_type = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            # print("pre position_bias: ", position_bias.shape)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                # print("mask", mask.shape)

                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

            # print(self.is_decoder)
            # print(attn_type)
            # print("hidden_states, key_states: ", hidden_states.shape, key_states.shape)
            # print("seq_length, real_seq_length, query_length: ", seq_length, real_seq_length, query_length)
            # print("position_bias: ", position_bias.shape)
            # print("score.shape", scores.shape)

            # print("query_states: ", query_states.shape)

            # print("==")

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class T5ForConditionalGenerationVanilla(nn.Module):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            if getattr(self, "teacher", None) is not None:
                
                loss = self.compute_distillation_loss(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,

                    lm_logits=lm_logits,
                    encoder_outputs=encoder_outputs,
                    decoder_outputs=decoder_outputs,
                )

            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # if not self.training:
            # print(torch.argmax(lm_logits, dim=-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def t5_modify_with_vanilla(transformer, petl_config):
    transformer.forward = functools.partial(T5ForConditionalGenerationVanilla.forward, transformer) # assign the self to the module (T5ForConditionalGeneration) object

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(".*SelfAttention|.*EncDecAttention", m_name):
            module.forward = functools.partial(T5AttentionVanilla.forward, module) # assign the self to the module (T5Attention) object

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
            self.enc_num_tokens = 0
            self.dec_num_tokens = 0
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

    model = t5_modify_with_vanilla(model, config)
    new_param = model.state_dict()
    # for i in new_param.keys():
    #     if "adapter" in i:
    #         print(i, new_param[i])

    # for sanity check
    if config.pbp_reduction_factor == 1:

        for p_name, param in model.named_parameters():
            if re.fullmatch(".*downsample.*|.*upsample.*", p_name):
                if p_name.endswith("weight"):
                    param.data = torch.eye(new_param[p_name].shape[0])
                elif p_name.endswith("bias"):
                    param.data.zero_()
                else:
                    raise NotImplementedError

        for p_name, param in model.named_parameters():
            if "side" in p_name:
                orig_name = p_name.replace(".side.", ".")

                assert torch.all(param == old_param[orig_name])

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
