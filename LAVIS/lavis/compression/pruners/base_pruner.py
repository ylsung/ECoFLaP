import torch
import torch.nn as nn

from time import time
from copy import deepcopy
from functools import partial

from lavis.datasets.data_utils import prepare_sample
from lavis.models.blip2_models.blip2_t5 import Blip2T5
from lavis.models.t5_models.t5 import T5
from lavis.models.clip_models.eva_model import EVA_CLIP
from lavis.compression.pruners.utils import (
    loss_vision_language, loss_language, loss_vision, print_time
)


class BasePruner:
    def __init__(self, 
                 model, 
                 data_loader, 
                 is_strct_pruning, 
                 keep_indices_or_masks_cache, 
                 importance_scores_cache, 
                 is_global, 
                 num_samples):

        self.model = model
        self.data_loader = data_loader
        self.is_strct_pruning = is_strct_pruning
        self.is_global = is_global
        self.num_samples = num_samples
        self.keep_indices_or_masks_cache = keep_indices_or_masks_cache
        self.importance_scores_cache = importance_scores_cache

    def compute_importance_scores(self, model, data_loader, loss_func):
        raise NotImplementedError

    def get_params(self, model):
        params = []
        names = []

        for name, param in model.named_parameters():
            names.append(name)
            params.append(param)

        return names, params

    def model_setup_and_record_attributes(self, model):
        dtype_record = {}
        requires_grad_record = {}
        for n, p in model.state_dict().items():
            dtype_record[n] = p.data.dtype
            p.data = p.data.type(torch.bfloat16)

        # set requires_grad to be true for getting model's derivatives
        for n, p in model.named_parameters():
            requires_grad_record[n] = p.requires_grad
            p.requires_grad = True

        device = list(self.model.parameters())[0].device
        # self.model.to("cpu")

        return dtype_record, requires_grad_record, device

    def model_reset(self, model, dtype_record, requires_grad_record, device):
        # set to original requires grad
        for n, p in model.named_parameters():
            p.requires_grad = requires_grad_record[n]

        for n, p in model.state_dict().items():
            p.data = p.data.type(dtype_record[n])
            
        model.to(device)

    def convert_spec_to_list(self, spec):
        num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = spec.split("-")

        num_layers = int(num_layers)
        res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = float(res_keep_ratio), float(attn_keep_ratio), float(ffn_keep_ratio)

        return num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio
    
    def create_pruned_arch(self, *args, **kwargs):
        return NotImplementedError
    
    @print_time
    def _prune(self, model, importance_scores, keep_indices_or_masks, prune_spec, ignore_layers, is_global):
        raise NotImplementedError
    
    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):
        raise NotImplementedError

# class BasePruner:
#     def __init__(
#         self,
#         model,
#         data_loader,
#         task_type="vision_language",
#         t5_prune_spec=None,
#         vit_prune_spec=None,
#         importance_scores_cache=None,
#         keep_indices_or_masks_cache=None,
#         is_strct_pruning=False,
#         num_samples=16,
#         is_global=False,
#     ):
#         self.model = model
#         self.data_loader = data_loader

#         self.is_strct_pruning = is_strct_pruning
#         self.task_type = task_type
        
#         self.keep_indices_or_masks_cache = keep_indices_or_masks_cache
#         self.importance_scores_cache = importance_scores_cache
#         self.t5_prune_spec = t5_prune_spec
#         self.vit_prune_spec = vit_prune_spec
#         self.is_global = is_global
#         self.num_samples = num_samples
        
#         if t5_prune_spec is not None and vit_prune_spec is not None:
#             self.prune_models = "t5+vit"
#         elif t5_prune_spec is not None:
#             self.prune_models = "t5"
#         else:
#             # vit_prune_spec is not None
#             self.prune_models = "vit"

#         self.t5_pruning_func = t5_strct_pruning if self.is_strct_pruning else t5_unstrct_pruning
#         self.vit_pruning_func = vit_strct_pruning if self.is_strct_pruning else vit_unstrct_pruning
        
#         if self.task_type == "vision_language":
#             assert isinstance(self.model, Blip2T5), f"the model should be Blip2T5 if using {self.task_type}"
#             self.loss_func = loss_vision_language
#         elif self.task_type == "language":
#             assert isinstance(self.model, T5), f"the model should be T5 if using {self.task_type}"
#             self.loss_func = loss_language
#         elif self.task_type == "vision":
#             assert isinstance(self.model, EVA_CLIP), f"the model should be EVA_CLIP if using {self.task_type}"
#             self.loss_func = loss_vision
#         else:
#             raise NotImplementedError(f"{self.task_type} is not supported")
        
#         self.t5_ignore_layers = [
#             "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
#             "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
#         ] # not used but may be used in the future
        
#         self.vit_ignore_layers = []
        
#         if isinstance(self.model, Blip2T5):
#             for k in self.model.t5_model.state_dict():
#                 # don't prune embedding layers and lm_head
#                 if any(sub_n in k for sub_n in ["shared", "embed_tokens", "lm_head", "layer_norm"]):
#                     self.t5_ignore_layers.append(k)
            
#             for k in self.model.visual_encoder.state_dict():
#                 # don't prune embedding layers and output layers
#                 if any(sub_n in k for sub_n in ["cls_token", "pos_embed", "patch_embed", "norm"]):
#                     self.vit_ignore_layers.append(k)

#         elif isinstance(self.model, T5):
#             for k in self.model.t5_model.state_dict():
#                 # don't prune embedding layers and lm_head
#                 if any(sub_n in k for sub_n in ["shared", "embed_tokens", "lm_head", "layer_norm"]):
#                     self.t5_ignore_layers.append(k)
                    
#         elif isinstance(self.model, EVA_CLIP):
#             for k in self.model.visual.state_dict():
#                 # don't prune embedding layers and output layers
#                 if any(sub_n in k for sub_n in ["cls_token", "pos_embed", "patch_embed", "norm"]):
#                     self.vit_ignore_layers.append(k)

#     def compute_sample_fisher(self, params, loss, return_outer_product=True):
#         ys = loss

#         grads = torch.autograd.grad(ys, params)  # first order gradient

#         assert len(grads) == len(params)

#         if not return_outer_product:
#             return grads
#         else:
#             return torch.outer(grads, grads)

#     def loss_vision_language(self, model, samples, cuda_enabled):
#         samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

#         # samples = {key: s.half() for key, s in samples.items()}

#         loss_dict = model(samples)
#         loss = loss_dict["loss"]

#         batch_len = len(samples["text_input"])

#         return loss, batch_len

#     def compute_importance_scores(self, model, data_loader, loss_func):
#         raise NotImplementedError

#     def unstrct_pruning(self, importance_measure, ratio):
#         masks = {}

#         for k, v in importance_measure.items():
#             top_k = int(v.numel() * ratio)

#             _, top_indices = v.float().reshape(-1).topk(top_k, dim=-1)

#             mask = torch.zeros((v.numel(),), dtype=bool, device=v.device) # 1D
#             mask.scatter_(-1, top_indices, 1)

#             mask = mask.reshape_as(v)

#             masks[k] = mask

#         return masks
    
#     def get_params(self, model):
#         params = []
#         names = []

#         for name, param in model.named_parameters():
#             names.append(name)
#             params.append(param)
            
#         return names, params

#     def model_setup_and_record_attributes(self, model):
#         dtype_record = {}
#         requires_grad_record = {}
#         for n, p in model.state_dict().items():
#             dtype_record[n] = p.data.dtype
#             p.data = p.data.type(torch.bfloat16)

#         # set requires_grad to be true for getting model's derivatives
#         for n, p in model.named_parameters():
#             requires_grad_record[n] = p.requires_grad
#             p.requires_grad = True
            
#         return dtype_record, requires_grad_record

#     def model_reset(self, model, dtype_record, requires_grad_record):
#         # set to original requires grad
#         for n, p in model.named_parameters():
#             p.requires_grad = requires_grad_record[n]

#         for n, p in model.state_dict().items():
#             p.data = p.data.type(dtype_record[n])

#     def reweighting_after_pruning(self, original_weights, keep_masks):
#         raise NotImplementedError

#     def read_cache(self, cache_file):
#         raise NotImplementedError

#     @print_time
#     def create_pruned_t5_arch(self, transformer, t5_prune_spec):
#         side_config = deepcopy(transformer.config)

#         num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = self.convert_spec_to_list(t5_prune_spec)
        
#         side_config.num_decoder_layers = num_layers
#         side_config.num_layers = num_layers

#         if self.is_strct_pruning:
#             # structural
#             side_config.d_model = get_pruned_dim(side_config.d_model, res_keep_ratio)
#             side_config.d_ff = get_pruned_dim(side_config.d_ff, ffn_keep_ratio)
#             side_config.d_kv = get_pruned_dim(side_config.d_kv, attn_keep_ratio)
#         else:
#             # unstructural
#             side_config.d_model = side_config.d_model
#             side_config.d_ff = side_config.d_ff
#             side_config.d_kv = side_config.d_kv
            
#         pruned_transformer = transformer.__class__(side_config)

#         return pruned_transformer

#     @print_time
#     def create_pruned_vit_arch(self, vit, vit_prune_spec):
#         num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = self.convert_spec_to_list(vit_prune_spec)
        
#         if self.is_strct_pruning:
#             pruned_vit = vit.__class__(
#                 img_size=vit.img_size,
#                 patch_size=vit.patch_size,
#                 use_mean_pooling=False,
#                 embed_dim=int(vit.embed_dim * res_keep_ratio),
#                 attn_dim=int(vit.attn_dim * attn_keep_ratio),
#                 depth=num_layers,
#                 num_heads=vit.num_heads,
#                 num_classes=vit.num_classes,
#                 mlp_ratio=vit.mlp_ratio * ffn_keep_ratio,
#                 qkv_bias=True,
#                 drop_path_rate=vit.drop_path_rate,
#                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                 use_checkpoint=vit.use_checkpoint,
#             )
#         else:
#             pruned_vit = vit.__class__(
#                 img_size=vit.img_size,
#                 patch_size=vit.patch_size,
#                 use_mean_pooling=False,
#                 embed_dim=vit.embed_dim,
#                 attn_dim=vit.attn_dim,
#                 depth=num_layers,
#                 num_heads=vit.num_heads,
#                 num_classes=vit.num_classes,
#                 mlp_ratio=vit.mlp_ratio,
#                 qkv_bias=True,
#                 drop_path_rate=vit.drop_path_rate,
#                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                 use_checkpoint=vit.use_checkpoint,
#             )
        
#         return pruned_vit

#     def convert_spec_to_list(self, spec):
#         num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = spec.split("-")

#         num_layers = int(num_layers)
#         res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = float(res_keep_ratio), float(attn_keep_ratio), float(ffn_keep_ratio)

#         return num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio

#     # def prune_t5(self, transformer, pruned_weights, keep_indices):
#     #     device = list(transformer.parameters())[0].device

#     #     transformer.to("cpu")

#     #     if self.t5_prune_spec is None:
#     #         # no prune spec
#     #         return transformer

#     #     pruned_transformer = self.create_pruned_t5_arch(transformer, self.t5_prune_spec)

#     #     pruned_transformer.load_state_dict(pruned_weights)

#     #     pruned_transformer.to(device)

#     #     return pruned_transformer
        
#     # def prune_vit(self, transformer, pruned_weights, keep_indices):
#     #     device = list(transformer.parameters())[0].device

#     #     transformer.to("cpu")

#     #     if self.vit_prune_spec is None:
#     #         # no prune spec
#     #         return transformer

#     #     pruned_transformer = self.create_pruned_vit_arch(transformer, self.vit_prune_spec)

#     #     pruned_transformer.load_state_dict(pruned_weights)

#     #     pruned_transformer.to(device)

#     #     return pruned_transformer

#     def t5_fill_missing_scores(self, transformer, scores):
#         for k, v in transformer.state_dict().items():
#             if k not in scores: # those are shared embeddings
#                 print(f"scores doesn't have {k}. Use shared.weight for it.")
#                 scores[k] = scores["shared.weight"]

#         return scores

#     def vit_fill_missing_scores(self, transformer, scores):
#         for k, v in transformer.state_dict().items():
#             if k not in scores:
#                 print(k, "not in scores, might cause errors in the following")
    
#         return scores
    
#     def fill_missing_scores(self, transformer, scores):
#         # some weights might not have gradients because they share weights with others
#         # so we need to manually assign their gradients
#         device = scores[list(scores.keys())[0]].device
        
#         for k, v in transformer.state_dict().items():
#             if k.startswith("t5_model"):
#                 if k not in scores: # those are shared embeddings
#                     print(f"scores doesn't have {k}. Use shared.weight for it.")
#                     scores[k] = scores["t5_model.shared.weight"]
#             elif k.startswith("visual_encoder"):
#                 # currently no exception needs to be tackled
#                 pass 
#             elif k.startswith("visual"):
#                 # currently no exception needs to be tackled
#                 pass
#             elif k.startswith("Qformer"):
#                 if k not in scores:
#                     scores[k] = torch.ones_like(v).to(device)
            
#         return scores
                
#     @print_time
#     def prune_t5(self, t5, t5_importance_scores, t5_keep_indices_or_masks, t5_prune_spec, t5_ignore_layers, is_global):
#         if t5_prune_spec is None:
#             # no prune spec
#             return t5
        
#         pruned_t5 = self.create_pruned_t5_arch(t5, t5_prune_spec)

#         _, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = self.convert_spec_to_list(t5_prune_spec)
        
#         # create keep indices and pruned_weights  
#         # if t5_keep_indices_or_masks is not None then the function will use
#         # the input t5_keep_indices_or_masks to get pruned_weights
#         pruned_t5_weights, t5_keep_indices_or_masks = self.t5_pruning_func(
#             t5, 
#             t5_importance_scores,
#             t5_keep_indices_or_masks,
#             res_keep_ratio, 
#             attn_keep_ratio, 
#             ffn_keep_ratio,
#             ignore_layers=t5_ignore_layers,
#             is_global=is_global,
#         )
        
#         pruned_t5.load_state_dict(pruned_t5_weights)
        
#         return pruned_t5, t5_keep_indices_or_masks

#     @print_time
#     def prune_vit(self, vit, vit_importance_scores, vit_keep_indices_or_masks, vit_prune_spec, vit_ignore_layers, is_global):
#         if vit_prune_spec is None:
#             # no prune spec
#             return vit

#         pruned_vit = self.create_pruned_vit_arch(vit, vit_prune_spec)
#         _, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = self.convert_spec_to_list(vit_prune_spec)

#         # if vit_keep_indices_or_masks is not None then the function will use
#         # the input vit_keep_indices_or_masks to get pruned_weights
#         pruned_vit_weights, vit_keep_indices_or_masks = self.vit_pruning_func(
#             vit, 
#             vit_importance_scores,
#             vit_keep_indices_or_masks,
#             res_keep_ratio, 
#             attn_keep_ratio, 
#             ffn_keep_ratio,
#             ignore_layers=vit_ignore_layers,
#             is_global=is_global,
#         )
        
#         pruned_vit.load_state_dict(pruned_vit_weights)
        
#         return pruned_vit, vit_keep_indices_or_masks

#     @print_time
#     def prune(self):
#         dtype_record, requires_grad_record = self.model_setup_and_record_attributes(self.model)

#         device = list(self.model.parameters())[0].device
#         self.model.to("cpu")

#         vit_importance_scores = None
#         t5_importance_scores = None
#         if self.importance_scores_cache is not None:
#             # cache = self.read_cache(self.importance_scores_cache)
#             importance_scores = torch.load(self.importance_scores_cache)
#             vit_importance_scores = importance_scores["vit"] if "vit" in importance_scores else None
#             t5_importance_scores = importance_scores["t5"] if "t5" in importance_scores else None
#         else:      
#             importance_scores = self.compute_importance_scores(self.model, self.data_loader, self.loss_func, device)

#             if isinstance(self.model, Blip2T5):
#                 vit_importance_scores = {k[15:]: v for k, v in importance_scores.items() if k.startswith("visual_encoder")} # filter out some info that is not for this transformer
#                 t5_importance_scores = {k[9:]: v for k, v in importance_scores.items() if k.startswith("t5_model")} # filter out some info that is not for this transformer
#             elif isinstance(self.model, T5):
#                 t5_importance_scores = {k[9:]: v for k, v in importance_scores.items() if k.startswith("t5_model")} # filter out some info that is not for this transformer
#             elif isinstance(self.model, EVA_CLIP):
#                 vit_importance_scores = {k[7:]: v for k, v in importance_scores.items() if k.startswith("visual")} # filter out some info that is not for this transformer

#         t5_keep_indices_or_masks, vit_keep_indices_or_masks = None, None
        
#         if self.keep_indices_or_masks_cache is not None:
#             keep_indices_or_masks = torch.load(self.keep_indices_or_masks_cache)
#             t5_keep_indices_or_masks = keep_indices_or_masks["t5"] if "t5" in keep_indices_or_masks else None
#             vit_keep_indices_or_masks = keep_indices_or_masks["vit"] if "vit" in keep_indices_or_masks else None

#         if self.prune_models == "t5+vit":
#             assert isinstance(self.model, Blip2T5)
#             assert vit_importance_scores is not None and t5_importance_scores is not None
            
#             # pruned the models
#             self.model.t5_model, t5_keep_indices_or_masks = self.prune_t5(
#                 self.model.t5_model, t5_importance_scores, t5_keep_indices_or_masks, self.t5_prune_spec, self.t5_ignore_layers, self.is_global
#             )

#             self.model.visual_encoder, vit_keep_indices_or_masks = self.prune_vit(
#                 self.model.visual_encoder, vit_importance_scores, vit_keep_indices_or_masks, self.vit_prune_spec, self.vit_ignore_layers, self.is_global
#             )

#         elif self.prune_models == "t5":
#             # if isinstance(self.model, Blip2T5):
#             self.model.t5_model, t5_keep_indices_or_masks = self.prune_t5(
#                 self.model.t5_model, t5_importance_scores, t5_keep_indices_or_masks, self.t5_prune_spec, self.t5_ignore_layers, self.is_global
#             )
#             # elif isinstance(self.model, T5):
#             #     self.model.t5_model, t5_keep_indices_or_masks = self.prune_t5(
#             #         self.model.t5_model, t5_importance_scores, self.t5_prune_spec, self.is_global
#             #     )
#         elif self.prune_models == "vit":
#             if isinstance(self.model, Blip2T5):
#                 self.model.visual_encoder, vit_keep_indices_or_masks = self.prune_vit(
#                     self.model.visual_encoder, vit_importance_scores, vit_keep_indices_or_masks, self.vit_prune_spec, self.vit_ignore_layers, self.is_global
#                 )
#             elif isinstance(self.model, EVA_CLIP):
#                 self.model.visual, vit_keep_indices_or_masks = self.prune_vit(
#                     self.model.visual, vit_importance_scores, vit_keep_indices_or_masks, self.vit_prune_spec, self.vit_ignore_layers, self.is_global
#                 )
    
#         if self.is_strct_pruning and isinstance(self.model, Blip2T5):
#             # only prune qformer when doing structural pruning
#             # because residual dimension might have changed 
#             self.model.Qformer, self.model.t5_proj = qformer_strct_pruning(
#                 self.model.Qformer, 
#                 self.model.t5_proj, 
#                 self.model.init_Qformer, 
#                 vit_keep_indices_or_masks["P_vit_res"] if vit_keep_indices_or_masks is not None else None, 
#                 t5_keep_indices_or_masks["P_res"] if t5_keep_indices_or_masks is not None else None
#             )

#         # if t5_keep_indices_or_masks is not None:
#         #     for _ in range(10):
#         #         total_value = 0 
#         #         for k, mask in t5_keep_indices_or_masks.items():
#         #             new_mask = torch.randn((mask.numel(),)).to(mask.device)
#         #             topk = int(mask.float().sum())
#         #             _, indices = torch.topk(new_mask, topk)
#         #             new_mask = torch.zeros_like(new_mask)
#         #             new_mask[indices] = 1
#         #             new_mask = new_mask.reshape_as(mask).to(mask.device)
#         #             total_value += (new_mask.float() * t5_importance_scores[k]).sum()
#         #         print(total_value)
    
#         #     total_value = 0
#         #     for k, mask in t5_keep_indices_or_masks.items():
#         #         total_value += (mask.float() * t5_importance_scores[k]).sum()
                
#         #     print(f"selected {total_value}")
            
#         keep_indices_or_masks_dict = {
#             "t5": t5_keep_indices_or_masks,
#             "vit": vit_keep_indices_or_masks,
#         }
        
#         # let the pruned model has the original
#         self.model_reset(self.model, dtype_record, requires_grad_record)
        
#         self.model.to(device)
        
#         return self.model, keep_indices_or_masks_dict
