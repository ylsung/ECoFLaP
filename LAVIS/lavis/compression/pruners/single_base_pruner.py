import torch
import torch.nn as nn

from time import time
from copy import deepcopy
from functools import partial

from lavis.datasets.data_utils import prepare_sample
from lavis.models.blip2_models.blip2_t5 import Blip2T5
from lavis.models.t5_models.t5 import T5
from lavis.models.clip_models.eva_model import EVA_CLIP
from lavis.compression.pruning.base_pruning import get_pruned_dim

from lavis.compression.pruning.t5_pruning import t5_strct_pruning, t5_unstrct_pruning
from lavis.compression.pruning.vit_pruning import vit_strct_pruning, vit_unstrct_pruning
from lavis.compression.pruning.qformer_pruning import qformer_strct_pruning
from lavis.compression.pruners.utils import (
    loss_vision_language, loss_language, loss_vision, print_time
)
from lavis.compression.pruners.base_pruner import BasePruner


class SingleBasePruner(BasePruner):
    def __init__(
        self,
        model,
        data_loader,
        prune_spec=None,
        importance_scores_cache=None,
        keep_indices_or_masks_cache=None,
        is_strct_pruning=False,
        num_samples=16,
        is_global=False,
        model_prefix="t5_model",
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_loader=data_loader,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
        )

        self.prune_spec = prune_spec
        self.model_prefix = model_prefix
        self.model_stem = eval(f"self.model.{model_prefix}") # self.model.t5_model, self.model.visual, etc
        
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
    def _prune(self, model, importance_scores, keep_indices_or_masks, prune_spec, ignore_layers, is_global, device):
        if prune_spec is None:
            # no prune spec
            return model
        
        # place to GPU
        importance_scores = {k: v.to(device) for k, v in importance_scores.items()}
        
        pruned = self.create_pruned_arch(model, prune_spec)

        _, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = self.convert_spec_to_list(prune_spec)
        
        # create keep indices and pruned_weights  
        # if keep_indices_or_masks is not None then the function will use
        # the input keep_indices_or_masks to get pruned_weights
        pruned_weights, keep_indices_or_masks = self.pruning_func(
            model, 
            importance_scores,
            keep_indices_or_masks,
            res_keep_ratio, 
            attn_keep_ratio, 
            ffn_keep_ratio,
            ignore_layers=ignore_layers,
            is_global=is_global,
        )
        
        pruned.load_state_dict(pruned_weights)
        
        return pruned, keep_indices_or_masks
    
    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):
        print("In: ", self.pruner_name)
        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)

        if keep_indices_or_masks is None:
            if self.keep_indices_or_masks_cache is not None:
                print("Use cached keep_indices_or_masks.")
                keep_indices_or_masks = torch.load(self.keep_indices_or_masks_cache)
        else:
            print("Use input keep_indices_or_masks.")

        if keep_indices_or_masks is None and importance_scores is None:
            # if there is no keep_indices_or_masks loaded, we need to compute importance_scores 
            # to determine keep_indices_or_masks later

            # if no input importance_scores
            if self.importance_scores_cache is not None:
                # cache = self.read_cache(self.importance_scores_cache)
                print("Use cached importance scores.")
                importance_scores = torch.load(self.importance_scores_cache)
            else:      
                importance_scores = self.compute_importance_scores(self.model, self.data_loader, self.loss_func, device)

                start_index = len(self.model_prefix) + 1
                importance_scores = {k[start_index:]: v for k, v in importance_scores.items() if k.startswith(self.model_prefix)} # filter out some info that is not for this transformer
        else:
            print("Use input importance scores.")

        pruned_model_stem, keep_indices_or_masks = self._prune(
            self.model_stem, importance_scores, keep_indices_or_masks, self.prune_spec, self.ignore_layers, self.is_global, device
        )
        
        # if keep_indices_or_masks is not None:
        #     for _ in range(10):
        #         total_value = 0 
        #         for k, mask in keep_indices_or_masks.items():
        #             topk = int(mask.float().sum())
        #             perm = torch.randperm(mask.numel())
        #             indices = perm[:topk]
        #             new_mask = torch.zeros((mask.numel(),), device=mask.device)
        #             new_mask[indices] = 1
        #             new_mask = new_mask.reshape_as(mask)
        #             total_value += (new_mask.float() * importance_scores[k].to(device)).sum()
        #         print(total_value)
    
        #     total_value = 0
        #     for k, mask in keep_indices_or_masks.items():
        #         total_value += (mask.float() * importance_scores[k].to(device)).sum()
                
        #     print(f"selected {total_value}")
        
        setattr(self.model, self.model_prefix, pruned_model_stem)

        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, keep_indices_or_masks


class T5BasePruner(SingleBasePruner):
    def __init__(
        self,
        model,
        data_loader,
        prune_spec=None,
        importance_scores_cache=None,
        keep_indices_or_masks_cache=None,
        is_strct_pruning=False,
        num_samples=16,
        is_global=False,
        model_prefix="t5_model",
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_loader=data_loader,
            prune_spec=prune_spec,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
            model_prefix=model_prefix,
        )
        self.pruning_func = t5_strct_pruning if self.is_strct_pruning else t5_unstrct_pruning
        
        self.loss_func = loss_language
        
        self.ignore_layers = [
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        ] # not used but may be used in the future
        for k in self.model_stem.state_dict():
            # don't prune embedding layers and lm_head
            if any(sub_n in k for sub_n in ["shared", "embed_tokens", "lm_head", "layer_norm"]):
                self.ignore_layers.append(k)

    def reweighting_after_pruning(self, original_weights, keep_masks):
        raise NotImplementedError

    def read_cache(self, cache_file):
        raise NotImplementedError

    @print_time
    def create_pruned_arch(self, transformer, prune_spec):
        side_config = deepcopy(transformer.config)

        num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = self.convert_spec_to_list(prune_spec)
        
        side_config.num_decoder_layers = num_layers
        side_config.num_layers = num_layers

        if self.is_strct_pruning:
            # structural
            side_config.d_model = get_pruned_dim(side_config.d_model, res_keep_ratio)
            side_config.d_ff = get_pruned_dim(side_config.d_ff, ffn_keep_ratio)
            side_config.d_kv = get_pruned_dim(side_config.d_kv, attn_keep_ratio)
        else:
            # unstructural
            side_config.d_model = side_config.d_model
            side_config.d_ff = side_config.d_ff
            side_config.d_kv = side_config.d_kv
            
        pruned_transformer = transformer.__class__(side_config)

        return pruned_transformer
    
    def fill_missing_scores(self, transformer, scores):
        # some weights might not have gradients because they share weights with others
        # so we need to manually assign their gradients
        device = scores[list(scores.keys())[0]].device
        
        for k, v in transformer.state_dict().items():
            if k.startswith("t5_model"):
                if k not in scores: # those are shared embeddings
                    print(f"scores doesn't have {k}. Use shared.weight for it.")
                    scores[k] = scores["t5_model.shared.weight"]

        return scores


class VITBasePruner(SingleBasePruner):
    def __init__(
        self,
        model,
        data_loader,
        prune_spec=None,
        importance_scores_cache=None,
        keep_indices_or_masks_cache=None,
        is_strct_pruning=False,
        num_samples=16,
        is_global=False,
        model_prefix="visual",
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_loader=data_loader,
            prune_spec=prune_spec,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
            model_prefix=model_prefix,
        )
        self.pruning_func = vit_strct_pruning if self.is_strct_pruning else vit_unstrct_pruning
        
        self.loss_func = loss_vision
        
        self.ignore_layers = []
        
        for k in self.model_stem.state_dict():
            # don't prune embedding layers and output layers
            if any(sub_n in k for sub_n in ["cls_token", "pos_embed", "patch_embed", "norm"]):
                self.ignore_layers.append(k)

    def reweighting_after_pruning(self, original_weights, keep_masks):
        raise NotImplementedError

    def read_cache(self, cache_file):
        raise NotImplementedError

    @print_time
    def create_pruned_arch(self, vit, vit_prune_spec):
        num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = self.convert_spec_to_list(vit_prune_spec)
        
        if self.is_strct_pruning:
            pruned_vit = vit.__class__(
                img_size=vit.img_size,
                patch_size=vit.patch_size,
                use_mean_pooling=False,
                embed_dim=int(vit.embed_dim * res_keep_ratio),
                attn_dim=int(vit.attn_dim * attn_keep_ratio),
                depth=num_layers,
                num_heads=vit.num_heads,
                num_classes=vit.num_classes,
                mlp_ratio=vit.mlp_ratio * ffn_keep_ratio,
                qkv_bias=True,
                drop_path_rate=vit.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                use_checkpoint=vit.use_checkpoint,
            )
        else:
            pruned_vit = vit.__class__(
                img_size=vit.img_size,
                patch_size=vit.patch_size,
                use_mean_pooling=False,
                embed_dim=vit.embed_dim,
                attn_dim=vit.attn_dim,
                depth=num_layers,
                num_heads=vit.num_heads,
                num_classes=vit.num_classes,
                mlp_ratio=vit.mlp_ratio,
                qkv_bias=True,
                drop_path_rate=vit.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                use_checkpoint=vit.use_checkpoint,
            )
        
        return pruned_vit
    
    def fill_missing_scores(self, transformer, scores):
        # some weights might not have gradients because they share weights with others
        # so we need to manually assign their gradients
        device = scores[list(scores.keys())[0]].device
        
        for k, v in transformer.state_dict().items():
            if k.startswith(self.model_prefix):
                if k not in scores: # those are shared embeddings
                    print(f"scores doesn't have {k}")

        return scores
