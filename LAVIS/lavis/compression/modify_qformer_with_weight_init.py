import torch
import torch.nn as nn
from lavis.compression.weight_matching import permutation_spec_from_axes_to_perm, apply_permutation


def get_qformer_ps(num_layers, cross_attention_freq):
    ps_dict = {
        "bert.embeddings.position_ids": (None, None),
        "bert.embeddings.LayerNorm.weight": (None, None),
        "bert.embeddings.LayerNorm.bias": (None,),
        **{f"bert.encoder.layer.{i}.attention.self.query.weight": (None, None) for i in range(num_layers)},                                                                                                
        **{f"bert.encoder.layer.{i}.attention.self.query.bias": (None,) for i in range(num_layers)},                                                                                                  
        **{f"bert.encoder.layer.{i}.attention.self.key.weight": (None, None) for i in range(num_layers)},                                                                                                  
        **{f"bert.encoder.layer.{i}.attention.self.key.bias": (None,) for i in range(num_layers)},                                                                                                    
        **{f"bert.encoder.layer.{i}.attention.self.value.weight": (None, None) for i in range(num_layers)},                                                                                                
        **{f"bert.encoder.layer.{i}.attention.self.value.bias": (None,) for i in range(num_layers)},                                                                                                  
        **{f"bert.encoder.layer.{i}.attention.output.dense.weight": (None, None) for i in range(num_layers)},                                                                                              
        **{f"bert.encoder.layer.{i}.attention.output.dense.bias": (None,) for i in range(num_layers)},                                                                                                
        **{f"bert.encoder.layer.{i}.attention.output.LayerNorm.weight": (None, None) for i in range(num_layers)},                                                                                          
        **{f"bert.encoder.layer.{i}.attention.output.LayerNorm.bias": (None,) for i in range(num_layers)},                                                                                            
        **{f"bert.encoder.layer.{i}.crossattention.self.query.weight": (None, None) for i in range(0, num_layers, cross_attention_freq)},                                                                                          
        **{f"bert.encoder.layer.{i}.crossattention.self.query.bias": (None,) for i in range(0, num_layers, cross_attention_freq)},                                                                                            
        **{f"bert.encoder.layer.{i}.crossattention.self.key.weight": (None, "P_vit_res") for i in range(0, num_layers, cross_attention_freq)},                                                                                            
        **{f"bert.encoder.layer.{i}.crossattention.self.key.bias": (None,) for i in range(0, num_layers, cross_attention_freq)},                                                                                              
        **{f"bert.encoder.layer.{i}.crossattention.self.value.weight": (None, "P_vit_res") for i in range(0, num_layers, cross_attention_freq)},                                                                                          
        **{f"bert.encoder.layer.{i}.crossattention.self.value.bias": (None,) for i in range(0, num_layers, cross_attention_freq)},                                                                                            
        **{f"bert.encoder.layer.{i}.crossattention.output.dense.weight": (None, None) for i in range(0, num_layers, cross_attention_freq)},                                                                                        
        **{f"bert.encoder.layer.{i}.crossattention.output.dense.bias": (None,) for i in range(0, num_layers, cross_attention_freq)},                                                                                          
        **{f"bert.encoder.layer.{i}.crossattention.output.LayerNorm.weight": (None, None) for i in range(0, num_layers, cross_attention_freq)},                                                                                    
        **{f"bert.encoder.layer.{i}.crossattention.output.LayerNorm.bias": (None,) for i in range(0, num_layers, cross_attention_freq)},                                                                                      
        **{f"bert.encoder.layer.{i}.intermediate_query.dense.weight": (None, None) for i in range(num_layers)},                                                                                           
        **{f"bert.encoder.layer.{i}.intermediate_query.dense.bias": (None,) for i in range(num_layers)},                                                                                             
        **{f"bert.encoder.layer.{i}.output_query.dense.weight": (None, None) for i in range(num_layers)},                                                                                                 
        **{f"bert.encoder.layer.{i}.output_query.dense.bias": (None,) for i in range(num_layers)},                                                                                                   
        **{f"bert.encoder.layer.{i}.output_query.LayerNorm.weight": (None, None) for i in range(num_layers)},
        **{f"bert.encoder.layer.{i}.output_query.LayerNorm.bias": (None,) for i in range(num_layers)},
    }

    return ps_dict


def get_t5_proj_ps():
    ps_dict = {
        "weight": ("P_t5_res", None),
        "bias": ("P_t5_res",),
    }

    return ps_dict


def qformer_pruning(orig_qformer, orig_t5_proj, qformer_init_func, P_vit_res, P_t5_res):
    device = list(orig_qformer.parameters())[0].device

    num_query_token = orig_qformer.config.query_length
    cross_attention_freq = orig_qformer.config.cross_attention_freq
    num_layers = orig_qformer.config.num_hidden_layers

    orig_qformer.to("cpu")
    orig_t5_proj.to("cpu")

    if P_vit_res is not None:
        distilled_vit_residual_dim = len(P_vit_res)

        distilled_Qformer, _ = qformer_init_func(
            num_query_token, 
            distilled_vit_residual_dim, 
            cross_attention_freq
        )
        distilled_Qformer.cls = None
        distilled_Qformer.bert.embeddings.word_embeddings = None
        distilled_Qformer.bert.embeddings.position_embeddings = None
        for layer in distilled_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        qformer_ps = permutation_spec_from_axes_to_perm(get_qformer_ps(num_layers, cross_attention_freq))
        distilled_qformer_weights = apply_permutation(qformer_ps, {"P_vit_res": P_vit_res}, orig_qformer.state_dict())
        distilled_Qformer.load_state_dict(distilled_qformer_weights)
    else:
        distilled_Qformer = orig_qformer

    if P_t5_res is not None:
        distilled_t5_residual_dim = len(P_t5_res)
        distilled_t5_proj = nn.Linear(
                distilled_Qformer.config.hidden_size, distilled_t5_residual_dim
            )
        t5_proj_ps = permutation_spec_from_axes_to_perm(get_t5_proj_ps())

        distilled_t5_proj_weights = apply_permutation(t5_proj_ps, {"P_t5_res": P_t5_res}, orig_t5_proj.state_dict())

        distilled_t5_proj.load_state_dict(distilled_t5_proj_weights)
    else:
        distilled_t5_proj = orig_t5_proj

    distilled_Qformer.to(device)
    distilled_t5_proj.to(device)

    return distilled_Qformer, distilled_t5_proj
