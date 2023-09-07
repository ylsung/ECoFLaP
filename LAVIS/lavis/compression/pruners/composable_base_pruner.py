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


class BLIPT5BasePruner(BasePruner):
    def __init__(
        self,
        model,
        data_loader,
        t5_prune_spec=None,
        vit_prune_spec=None,
        t5_pruning_method=None,
        vit_pruning_method=None,
        t5_importance_scores_cache=None,
        t5_keep_indices_or_masks_cache=None,
        vit_importance_scores_cache=None,
        vit_keep_indices_or_masks_cache=None,
        importance_scores_cache=None,
        keep_indices_or_masks_cache=None,
        is_strct_pruning=False,
        num_samples=16,
        is_global=False,
        t5_model_prefix="t5_model",
        vit_model_prefix="visual_encoder",
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
        
        self.t5_prune_spec = t5_prune_spec
        self.vit_prune_spec = vit_prune_spec
        
        assert t5_pruning_method is not None
        assert vit_pruning_method is not None
        
        self.t5_model_prefix = t5_model_prefix
        self.vit_model_prefix = vit_model_prefix
        
        from lavis.compression import load_pruner

        t5_config = {
            "prune_spec": t5_prune_spec,
            "importance_scores_cache": t5_importance_scores_cache,
            "keep_indices_cache": t5_keep_indices_or_masks_cache,
            "is_strct_pruning": is_strct_pruning,
            "is_global": is_global,
            "model_prefix": self.t5_model_prefix,
        }
        self.t5_pruner = load_pruner(t5_pruning_method, model, data_loader, cfg=t5_config)

        vit_config = {
            "prune_spec": vit_prune_spec,
            "importance_scores_cache": vit_importance_scores_cache,
            "keep_indices_cache": vit_keep_indices_or_masks_cache,
            "is_strct_pruning": is_strct_pruning,
            "is_global": is_global,
            "model_prefix": self.vit_model_prefix,
        }
        self.vit_pruner = load_pruner(vit_pruning_method, model, data_loader, cfg=vit_config)
        
        self.loss_func = loss_vision_language
        
    def fill_missing_scores(self, transformer, scores):
        # some weights might not have gradients because they share weights with others
        # so we need to manually assign their gradients
        device = scores[list(scores.keys())[0]].device
        
        for k, v in transformer.state_dict().items():
            if k.startswith("t5_model"):
                if k not in scores: # those are shared embeddings
                    print(f"scores doesn't have {k}. Use shared.weight for it.")
                    scores[k] = scores["t5_model.shared.weight"]
            elif k.startswith("visual_encoder"):
                # currently no exception needs to be tackled
                pass 
            elif k.startswith("visual"):
                # currently no exception needs to be tackled
                pass
            elif k.startswith("Qformer"):
                if k not in scores:
                    scores[k] = torch.ones_like(v).to(device)
            
        return scores

    @print_time
    def prune(self):
        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)

        t5_keep_indices_or_masks, vit_keep_indices_or_masks = None, None
        if self.keep_indices_or_masks_cache is not None:
            keep_indices_or_masks = torch.load(self.keep_indices_or_masks_cache)
            t5_keep_indices_or_masks = keep_indices_or_masks["t5"] if "t5" in keep_indices_or_masks else None
            vit_keep_indices_or_masks = keep_indices_or_masks["vit"] if "vit" in keep_indices_or_masks else None
            
            del keep_indices_or_masks

        vit_importance_scores, t5_importance_scores = None, None
        if t5_keep_indices_or_masks is None or vit_keep_indices_or_masks is None:
            # if there is no keep_indices_or_masks loaded, we need to compute importance_scores 
            # to determine keep_indices_or_masks later

            # if no input importance_scores
            if self.importance_scores_cache is not None:
                # cache = self.read_cache(self.importance_scores_cache)
                importance_scores = torch.load(self.importance_scores_cache)
                vit_importance_scores = importance_scores["vit"] if "vit" in importance_scores else None
                t5_importance_scores = importance_scores["t5"] if "t5" in importance_scores else None
            else:
                importance_scores = self.compute_importance_scores(self.model, self.data_loader, self.loss_func, device)
                t5_start_index = len(self.t5_model_prefix) + 1
                t5_importance_scores = {k[t5_start_index:]: v for k, v in importance_scores.items() if k.startswith(self.t5_model_prefix)} # filter out some info that is not for this transformer
                # t5_importance_scores = {k: v for k, v in importance_scores.items() if k.startswith(self.t5_model_prefix)} # filter out some info that is not for this transformer

                vit_start_index = len(self.vit_model_prefix) + 1
                vit_importance_scores = {k[vit_start_index:]: v for k, v in importance_scores.items() if k.startswith(self.vit_model_prefix)} # filter out some info that is not for this transformer
                # vit_importance_scores = {k: v for k, v in importance_scores.items() if k.startswith(self.vit_model_prefix)} # filter out some info that is not for this transformer

            del importance_scores
            # place to GPU
            # t5_importance_scores = {k: v.to(device) for k, v in t5_importance_scores.items()}
            # vit_importance_scores = {k: v.to(device) for k, v in vit_importance_scores.items()}

        self.model, t5_keep_indices_or_masks = self.t5_pruner.prune(
            importance_scores=t5_importance_scores, keep_indices_or_masks=t5_keep_indices_or_masks
        )
        
        self.model, vit_keep_indices_or_masks = self.vit_pruner.prune(
            importance_scores=vit_importance_scores, keep_indices_or_masks=vit_keep_indices_or_masks
        )

        if self.is_strct_pruning:
            # only prune qformer when doing structural pruning
            # because residual dimension might have changed 
            self.model.Qformer, self.model.t5_proj = qformer_strct_pruning(
                self.model.Qformer, 
                self.model.t5_proj, 
                self.model.init_Qformer, 
                vit_keep_indices_or_masks["P_vit_res"] if vit_keep_indices_or_masks is not None else None, 
                t5_keep_indices_or_masks["P_res"] if t5_keep_indices_or_masks is not None else None
            )

        keep_indices_or_masks_dict = {
            "t5": t5_keep_indices_or_masks,
            "vit": vit_keep_indices_or_masks,
        }
        
        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, keep_indices_or_masks_dict