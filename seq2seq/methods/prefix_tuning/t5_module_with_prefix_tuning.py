import transformers
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from typing import Optional, Tuple, Union, Dict, Any

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, BaseModelOutput


def sample_embed(embed, sample_size, start_idx, end_idx):
    # borrowed from https://github.com/r-three/t-few/blob/4e581fa0b8f53e36da252a15bd581d365d4dd333/src/models/prompt_tuning.py
    embed_weight = embed.weight
    rand_idx = torch.randint(start_idx, end_idx, (sample_size,))
    return embed_weight[rand_idx].detach()


class T5AttentionPrefixWrapper(nn.Module):
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
        decoder_first_token_or_encoder=False,
        prefix=None,
    ): 
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        real_seq_length_including_prefix = seq_length + self.prefix_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

            # print("query_length, {}, past length {}, seq_length {}".format(query_length, past_key_value[0].shape[2], seq_length))

        # 3 cases
        # 1. prefix=None, past_key_value=None, key_value_states=None <- prefix_length = 0 for encoder or decoder
        # 2. prefix=None, past_key_value!=None, key_value_states=None <- when decoding > 1st token
        # 3. prefix!=None, past_key_value=None, key_value_states=None <- prefix_length != 0 for encoder or decoding the 1st token or in training

        # key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        # if prefix is not None and past_key_value is None: # 3rd case
        #     key_length += self.prefix_length # the prefix is added to encoder_outputs and it is not considered in past_key_value

        # print("key_length:", key_length)

        # if self.prefix_in_enc_dec:
        #     if decoder_first_token_or_encoder:
        #         key_length = past_key_value[0].shape[2] + key_value_states.shape[1]
        #     else:
        #         key_length = past_key_value[0].shape[2]

        # print("After key length: ", key_length)

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value, prefix_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
                attn_type = "self-attn-1"
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))
                attn_type = "cross-attn-1"

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                    attn_type = "self-attn-2"
                else:
                    # cross-attn
                    hidden_states = past_key_value
                    attn_type = "cross-attn-2"

                    # if self.prefix_in_enc_dec and decoder_first_token_or_encoder:
                    #     hidden_states = torch.cat([past_key_value, shape(proj_layer(key_value_states))], dim=2) # concat prefix and encoder-decoder outputs

            if prefix_key_value is not None:
                hidden_states = torch.cat([prefix_key_value, hidden_states], dim=2)
            
            return hidden_states, attn_type

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states, attn_type = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None,
            prefix[0] if prefix is not None else None,
        )
        value_states, attn_type = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None,
            prefix[1] if prefix is not None else None,
        )

        key_length = key_states.shape[2]

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length + self.prefix_length if past_key_value is None else real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True

                # print("zero position bias")
            else:
                if past_key_value is None:
                    position_bias = self.compute_bias(real_seq_length + self.prefix_length, key_length, device=scores.device)
                    # print("In past_key_value is None, encoder or decoder = 0")
                else:
                    # real_seg_length already contains prefix_length
                    position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)
                    # print("decoder > 0")

            # print("pre prefix position_bias: ", position_bias.shape)

            position_bias = position_bias[:, :, self.prefix_length:, :]
            
            # print(attn_type)

            # if not self.training:
            #     print(attn_type)
            #     print(torch.sum(key_states.transpose(1, 2).reshape(batch_size, key_length, -1).abs(), dim=-1))

            # print("pre position_bias: ", position_bias.shape)

            # print("position_bias", position_bias.shape)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                # print("hidden_states", hidden_states.shape)
                # print("position_bias and mask", position_bias.shape, mask.shape)

                # we don't pad when self-attn 2 because the padded mask created in self-attn 1 will be passed to self-attn 2
                # print("pre mask", mask.shape)
                if attn_type in ["self-attn-1", "cross-attn-1", "cross-attn-2"]:
                    mask = torch.nn.functional.pad(mask, value=-float("0"), pad=(self.prefix_length, 0))
     
                # print("padded mask", mask.shape)

                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

            # print(self.is_decoder)
            # print("hidden_states, key_states: ", hidden_states.shape, key_states.shape)
            # print("seq_length, real_seq_length, query_length, real_seq_length_including_prefix: ", seq_length, real_seq_length, query_length, real_seq_length_including_prefix)
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
        
        # if not self.training:
        #     print(torch.argmax(attn_weights, dim=-1))

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


class T5LayerSelfAttentionPrefixWrapper(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        decoder_first_token_or_encoder=False,
        prefix=None,
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
            decoder_first_token_or_encoder=decoder_first_token_or_encoder,
            prefix=prefix,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttentionPrefixWrapper(nn.Module):
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
        decoder_first_token_or_encoder=False,
        prefix=None,
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
            decoder_first_token_or_encoder=decoder_first_token_or_encoder,
            prefix=prefix,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5BlockPrefixWrapper(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        decoder_first_token_or_encoder=False,
        prefix=None,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        if prefix is not None:
            self_attn_prefix = prefix[:2]
            cross_attn_prefix = prefix[2:]
        else:
            self_attn_prefix, cross_attn_prefix = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            decoder_first_token_or_encoder=decoder_first_token_or_encoder,
            prefix=self_attn_prefix,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                decoder_first_token_or_encoder=decoder_first_token_or_encoder,
                prefix=cross_attn_prefix,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5StackPrefixWrapper(nn.Module):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decoder_first_token_or_encoder=False,
        prefixes=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        # if prefixes is not None:
        #     # the attention mask is created in tokenizer and which doesn't consider prefix
        #     prefix_mask = torch.ones(batch_size, prefixes[0][0].shape[2], device=inputs_embeds.device)
        #     attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)

        # if self.is_decoder and encoder_attention_mask is not None:
        #     # when decoding, the mask is reused, but it didn't consider the prefix's mask

        #     # only in the side decoder 
        #     # if prefixes is not None:
                
        #     prefix_mask = torch.ones(
        #         batch_size, self.prefix_length, device=inputs_embeds.device, dtype=torch.long
        #     )

        #     encoder_attention_mask = torch.cat([prefix_mask, encoder_attention_mask], dim=-1)

            # print("First", encoder_attention_mask.shape)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]

            # having prefix for cross-attn
            # if prefixes is not None:
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

            # prefix_mask = torch.ones(
            #     batch_size, self.prefix_length, device=inputs_embeds.device, dtype=torch.long
            # )

            # encoder_attention_mask = torch.cat([prefix_mask, encoder_attention_mask], dim=-1)
            # print("Second", encoder_attention_mask.shape)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if prefixes is None:
            prefixes = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()

            # having prefix for cross-attn
            # if prefixes is not None:

            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)

            # prefix_mask = torch.ones(
            #     batch_size, self.prefix_length, device=inputs_embeds.device, dtype=torch.long
            # )

            # encoder_attention_mask = torch.cat([prefix_mask, encoder_attention_mask], dim=-1)
    

            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

            # print("Extended: ", encoder_extended_attention_mask.shape)
        else:
            encoder_extended_attention_mask = None


        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            prefix = prefixes[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    decoder_first_token_or_encoder,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    decoder_first_token_or_encoder=decoder_first_token_or_encoder,
                    prefix=prefix,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class T5StackForPrefix(nn.Module):

    def __init__(self, T5_backbone, config, petl_config):
        super().__init__()

        self.config = config

        self.backbone = T5_backbone

        self.is_decoder = self.backbone.is_decoder

        if self.is_decoder:
            # self.hard_prompt = None
            self.register_buffer("hard_prompt", None)

            self.prompts = self.generate_trainable_prompts_embeddings(petl_config.dec_num_tokens, config.d_model, petl_config.init_from_emb) # batch size is 1
            shared_layer = nn.Linear(config.d_model, petl_config.prefix_dim)
            self.embed_to_kv = nn.ModuleList(
                [self.construct_proj_layer(config.d_model, petl_config.prefix_dim, config.d_model * 4, shared_first_layer=shared_layer) for _ in range(config.num_decoder_layers)]
            ) # for self-attn and cross-attn

            self.set_prefix_length(self.prompts.shape[1])
        else:
            # self.hard_prompt = None
            self.register_buffer("hard_prompt", None)
            self.prompts = self.generate_trainable_prompts_embeddings(petl_config.enc_num_tokens, config.d_model, petl_config.init_from_emb) # batch size is 1
            shared_layer = nn.Linear(config.d_model, petl_config.prefix_dim)
            self.embed_to_kv = nn.ModuleList(
                [self.construct_proj_layer(config.d_model, petl_config.prefix_dim, config.d_model * 2, shared_first_layer=shared_layer) for _ in range(config.num_layers)]
            ) # for self-attn and cross-attn

            self.set_prefix_length(self.prompts.shape[1])

        self.main_input_name = self.backbone.main_input_name
        self.prompts_expand_after = petl_config.prompts_expand_after # save memory for training and inference but doesn't work for when using beam search

    def construct_proj_layer(self, input_dim, mid_dim, output_dim, shared_first_layer=None):
        if shared_first_layer is None:
            first_layer = nn.Linear(input_dim, mid_dim)
        else:
            first_layer = shared_first_layer
            
        return nn.Sequential(
            first_layer,
            nn.Tanh(),
            nn.Linear(mid_dim, output_dim)
        )

    def generate_trainable_prompts_embeddings(self, num_tokens, d_model, init_from_emb=False):
        if not init_from_emb:
            return nn.Parameter(torch.randn(1, num_tokens, d_model))

        return nn.Parameter(
                sample_embed(
                    embed=self.backbone.embed_tokens,
                    sample_size=num_tokens,
                    start_idx=3,
                    end_idx=5003,
                    ).unsqueeze(0)
            )

    def set_hard_prompts(self, hard_prompt):
        self.register_buffer('hard_prompt', hard_prompt)

        self.set_prefix_length(self.hard_prompt.shape[1])

    def set_prefix_length(self, prefix_length):
        for module in self.modules():
            if isinstance(module, nn.Module):
                module.prefix_length = prefix_length

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        prefixes=None,
    ):

        prefix_hidden_states = None
        decoder_first_token_or_encoder = False

        # generate the prefix from the backbone network
        if prefixes is None and past_key_values is None:
            # the encoder or when generate the first token by the decoder
            B = input_ids.shape[0]

            if self.hard_prompt is not None:
                prompts = self.backbone.embed_tokens(self.hard_prompt)
            else:
                prompts = self.prompts

                # prompts = self.backbone.embed_tokens(input_ids)

                # self.set_prefix_length(prompts.shape[1])

            prefixes = []

            for embed_to_kv_layer in self.embed_to_kv:
                
                chunks = 2 if not self.is_decoder else 4

                L = prompts.shape[1]
                # h = embed_to_kv_layer(h) # (1, L, 2D) or (1, L, 4D)

                h = embed_to_kv_layer(prompts)

                if self.prompts_expand_after:
                    h = h.expand(B, -1, -1) # (B, L, 2D)
                h = h.reshape(B, L, self.config.num_heads, h.shape[-1] // self.config.num_heads) # (B, L, H, 2 * D_per_head)
                h = h.transpose(1, 2) # (B, H, L, 2 D_per_head)

                # h = torch.zeros_like(h)
                
                kv_pairs = h.chunk(dim=-1, chunks=chunks) # each has (B, H, L, D_per_head)

                prefixes.append(kv_pairs)

            prefixes = tuple(prefixes)

            decoder_first_token_or_encoder = True

        # Insert the prefix
        outputs = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            prefixes=prefixes,
        )

        return outputs
