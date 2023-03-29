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


class T5ForConditionalGenerationDistillation(nn.Module):
    def compute_distillation_loss(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        lm_logits=None,
        encoder_outputs=None,
        decoder_outputs=None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        # label loss
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss_label = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))


        loss_cos = 0
        loss_kl = 0

        with torch.no_grad():
            teacher_output = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=None,
                use_cache=use_cache,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=return_dict,
            )

        # teacher CE loss
        s_logits = lm_logits
        t_logits = teacher_output["logits"]

        if self.alpha_kl > 0:
            if self.restrict_kl_to_mask:
                mask = (labels > -1).unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
                
                # print(t_logits.shape, mask.shape)

                s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
                s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
                t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
                t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
                assert t_logits_slct.size() == s_logits_slct.size()
            else:
                t_logits_slct = t_logits
                s_logits_slct = s_logits

            ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
    
            loss_kl = (
                ce_loss_fct(
                    nn.functional.log_softmax(s_logits_slct / self.temperature, dim=-1),
                    nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
                )
                * (self.temperature) ** 2
            )

        if self.alpha_cos > 0:
            cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

            def hidden_states_cosine_loss(s_hidden_states, t_hidden_states, mask):
                dim = s_hidden_states.size(-1)

                if mask is not None:
            
                    mask = mask.unsqueeze(-1).expand_as(s_hidden_states)  # (bs, seq_length, dim)

                    assert s_hidden_states.size() == t_hidden_states.size()

                    s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
                    t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
                else:
                    s_hidden_states_slct = s_hidden_states
                    t_hidden_states_slct = t_hidden_states

                s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
                t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

                target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
                loss_cos = cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
                return loss_cos
            # loss += self.alpha_cos * loss_cos

            for s_hidden_states, t_hidden_states in zip(encoder_outputs.hidden_states, teacher_output["encoder_hidden_states"]):
                loss_cos_layer = hidden_states_cosine_loss(s_hidden_states, t_hidden_states, attention_mask)

                loss_cos = loss_cos + 1 / len(encoder_outputs.hidden_states) * loss_cos_layer

            decoder_mask = (labels > -1).unsqueeze(-1).expand_as(s_logits)
            for s_hidden_states, t_hidden_states in zip(decoder_outputs.hidden_states, teacher_output["decoder_hidden_states"]):
                # print(s_hidden_states.shape, t_hidden_states.shape, decoder_mask.shape)
                # print(decoder_mask[:, :, 0])
                loss_cos_layer = hidden_states_cosine_loss(s_hidden_states, t_hidden_states, decoder_mask[:, :, 0])

                loss_cos = loss_cos + 1 / len(decoder_outputs.hidden_states) * loss_cos_layer

        # if self.training:
        print(loss_label, loss_kl, loss_cos)

        return self.alpha_label * loss_label + self.alpha_kl * loss_kl + self.alpha_cos * loss_cos


def t5_modify_with_distillation(transformer, teacher_model, petl_config):
    transformer.compute_distillation_loss = functools.partial(T5ForConditionalGenerationDistillation.compute_distillation_loss, transformer) # assign the self to the module (T5ForConditionalGeneration) object

    # assign teacher objects and alpha

    transformer.config.output_attentions = True
    transformer.encoder.config.output_attentions = True
    transformer.decoder.config.output_attentions = True

    transformer.config.output_hidden_states = True
    transformer.encoder.config.output_hidden_states = True
    transformer.decoder.config.output_hidden_states = True

    transformer.teacher = teacher_model
    transformer.temperature = petl_config.temperature
    transformer.alpha_label = petl_config.alpha_label
    transformer.alpha_kl = petl_config.alpha_kl
    transformer.alpha_cos = petl_config.alpha_cos
    transformer.restrict_kl_to_mask = petl_config.restrict_kl_to_mask

    return transformer


if __name__ == "__main__":
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import argparse

    from copy import deepcopy

    from seq2seq.methods.vanilla.modify_model_with_vanilla import t5_modify_with_vanilla

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
            self.alpha_label = 0.33
            self.alpha_kl = 0.33
            self.alpha_cos = 0.33
            self.restrict_kl_to_mask = True
            self.temperature = 0.1

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

    model =  t5_modify_with_distillation(model, deepcopy(model), config)

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
