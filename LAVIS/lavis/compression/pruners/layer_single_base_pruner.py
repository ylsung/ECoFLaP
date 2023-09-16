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


class LayerWiseBasePruner(BasePruner):
    def __init__(
        self,
        model,
        data_loader,
        prune_spec=None,
        importance_scores_cache=None,
        keep_indices_or_masks_cache=None,
        is_strct_pruning=False,
        num_samples=64,
        is_global=False,
        model_prefix="t5_model",
        sparsity_ratio_granularity=None,
        max_sparsity_per_layer=0.8,
        score_method="obd",
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

        self.sparsity_ratio_granularity = sparsity_ratio_granularity
        self.max_sparsity_per_layer = max_sparsity_per_layer
        self.score_method = score_method

        self.prune_spec = prune_spec
        self.model_prefix = model_prefix
        self.prune_n = 0
        self.prune_m = 0
        self.model_stem = getattr(self.model, model_prefix, None) # self.model.t5_model, self.model.visual, etc
        
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
        # for n, p in model.state_dict().items():
        for n, p in model.named_parameters():
            dtype_record[n] = p.data.dtype
            # p.data = p.data.type(torch.bfloat16)

        # set requires_grad to be true for getting model's derivatives
        for n, p in model.named_parameters():
            requires_grad_record[n] = p.requires_grad
            p.requires_grad = True

        device = next(iter(model.parameters())).device
        # self.model.to("cpu")

        return dtype_record, requires_grad_record, device

    def model_reset(self, model, dtype_record, requires_grad_record, device):
        # set to original requires grad
        for n, p in model.named_parameters():
            p.requires_grad = requires_grad_record[n]

        # for n, p in model.state_dict().items():
        for n, p in model.named_parameters():
            p.data = p.data.type(dtype_record[n])
            
        model.to(device)
            
    def convert_spec_to_list(self, spec):
        num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = spec.split("-")

        num_layers = int(num_layers)
        res_keep_ratio, attn_keep_ratio, ffn_keep_ratio = float(res_keep_ratio), float(attn_keep_ratio), float(ffn_keep_ratio)

        return num_layers, res_keep_ratio, attn_keep_ratio, ffn_keep_ratio
    
    def create_pruned_arch(self, *args, **kwargs):
        return NotImplementedError


class LayerSparsity:
    def __init__(self, model, data_loader, loss_func, num_samples, original_sparsity, max_sparsity_per_layer=0.8, score_method="obd", layer_to_group_mapping={}):
        self.importance_measure = {}
        self.model = model
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.num_samples = num_samples
        self.original_sparsity = original_sparsity
        self.layer_to_group_mapping = layer_to_group_mapping
        self.max_sparsity_per_layer = max_sparsity_per_layer
        
        self.score_method = score_method
        
        self.score_compute, self.score_aggregate = score_method.split("_")
        
        assert self.max_sparsity_per_layer >= self.original_sparsity
    
    @print_time
    def return_sparsity(self):
        original_sparsity = self.original_sparsity
        layer_to_group_mapping = self.layer_to_group_mapping

        if layer_to_group_mapping is None or len(layer_to_group_mapping) == 0:
            class uniform_sparsity_module:
                def __getitem__(self, key):
                    return original_sparsity
            return uniform_sparsity_module()

        # compute the global information
        if len(self.importance_measure) == 0:
            self.importance_measure = self.compute_importance_scores()
        
        # create the layer list that for each group
        group_to_layer_mapping = {}
        for k, v in layer_to_group_mapping.items():
            if v not in group_to_layer_mapping:
                group_to_layer_mapping[v] = []

            group_to_layer_mapping[v].append(k)
        
        # store the num of parameters for each group and the total paramters
        num_parameters_dict = {}
        total_parameters = 0
        for k, v in self.model.named_parameters():
            if k in layer_to_group_mapping:
                num_parameters_dict[k] = v.numel()
                total_parameters += v.numel()
        
        # total params to keep
        total_parameters_to_keep = total_parameters * (1 - original_sparsity)
        
        # store the importance per parameter for each group
        group_scores = {}
        group_num_parameters = {}
        for group_name, layers in group_to_layer_mapping.items():
            if group_name not in group_scores:
                group_scores[group_name] = 0
            
            num_params = 0
            for l in layers:
                group_scores[group_name] += self.importance_measure[l].sum()
                
                num_params += num_parameters_dict[l]
            
            if self.score_aggregate == "avg":
                group_scores[group_name] /= num_params # normalization
            
            group_num_parameters[group_name] = num_params
            
        # def normalize(group_scores, group_num_parameters):
        #     scores = list(group_scores.values())
            
        #     normalized_factor = sum(group_scores.values())
            
        #     print("before: ", normalized_factor)
            
        #     # mean = torch.mean(torch.FloatTensor(scores))
        #     # std = torch.std(torch.FloatTensor(scores))
            
        #     # return {group_name: torch.clamp(group_score, min=mean-std, max=mean+std) for group_name, group_score in group_scores.items()}
            
        #     return {group_name: torch.clamp(group_score, max=normalized_factor*group_num_parameters[group_name]/total_parameters_to_keep) for group_name, group_score in group_scores.items()}
        
        # group_scores = normalize(group_scores, group_num_parameters)
        # # the total score sum over all groups
        # normalized_factor_for_score = sum(group_scores.values())
        
        # print("after: ", normalized_factor_for_score)
 
        # def convert_score_to_sparsity(group_name, group_score, group_param):
        #     keep_param = total_parameters_to_keep * (group_score / normalized_factor_for_score)
            
        #     # print(group_name, keep_param)
            
        #     if keep_param > group_param:
        #         print(group_name, keep_param, group_param)
            
        #     keep_param = min(group_param, keep_param)
            
        #     sparsity = 1 - keep_param / group_param
            
        #     return sparsity
        
        # group_sparsity = {
        #     group_name: convert_score_to_sparsity(group_name, group_score, group_num_parameters[group_name])
        #     for group_name, group_score in group_scores.items()
        # }
        
        def compute_the_sparsity_per_group(total_parameters_to_keep, group_scores, group_num_parameters, max_sparsity_per_layer=0.8):
            scores = torch.FloatTensor(list(group_scores.values()))
            num_parameters = torch.LongTensor(list(group_num_parameters.values()))
            
            parameters_to_keep_per_group = torch.zeros_like(scores, dtype=int)
            
            parameters_to_keep_per_group += torch.ceil(num_parameters * (1 - max_sparsity_per_layer)).int() # to gaurantee the max_sparsity
            
            while parameters_to_keep_per_group.sum() < total_parameters_to_keep:
                total_ratio = torch.sum(scores)
                
                rest_total_parameters_to_keep = total_parameters_to_keep - parameters_to_keep_per_group.sum()
                
                parameters_to_add = torch.ceil((scores / total_ratio) * rest_total_parameters_to_keep)
                
                parameters_to_keep_per_group = parameters_to_keep_per_group + parameters_to_add
                
                scores[parameters_to_keep_per_group >= num_parameters] = 0 # make sure they are not going to add more parameters
                
                parameters_to_keep_per_group = torch.clamp(parameters_to_keep_per_group, max=num_parameters) # remove the extra parameters

                # the following codes are optional
                # they are to make sure the sum of parameters_to_keep_per_group is EXACTLY the same as total_parameters_to_keep
                if parameters_to_add.sum() == 0: # for some reason the algo cannot add more parameters
                    # the algo stuck
                    current_sum = parameters_to_keep_per_group.sum()
                    if current_sum < total_parameters_to_keep:
                        num_need_to_add = total_parameters_to_keep - current_sum
                        
                        while num_need_to_add > 0:
                            # distributed the parameters to the rest of groups
                            for index in torch.where(scores > 0)[0]:
                                parameters_can_add = min(
                                    num_need_to_add, num_parameters[index] - parameters_to_keep_per_group[index]
                                )
                                parameters_to_keep_per_group[index] += parameters_can_add
                                
                                num_need_to_add -= parameters_can_add
                                
                                if num_need_to_add == 0:
                                    break
                                
                if parameters_to_keep_per_group.sum() > total_parameters_to_keep: # for some reason the algo cannot add more parameters
                    # the algo stuck
                    current_sum = parameters_to_keep_per_group.sum()

                    num_need_to_remove = current_sum - total_parameters_to_keep
                    
                    while num_need_to_remove > 0:
                        # remove the parameters from full groups
                        for index in torch.argsort(parameters_to_keep_per_group, descending=True, stable=True):
                            parameters_can_remove = min(
                                num_need_to_remove, 
                                parameters_to_keep_per_group[index] - (num_parameters[index] * (1 - max_sparsity_per_layer)).int() # extra parameters
                            )
                            parameters_to_keep_per_group[index] += parameters_can_remove
                            
                            num_need_to_remove -= parameters_can_remove
                            
                            if num_need_to_remove == 0:
                                break
                            
                ############################### Optional codes end here
            
            # convert the group parameters to keep to sparsity    
            group_sparsity = {}
            
            for k, param_to_keep, group_max_param in zip(group_num_parameters.keys(), parameters_to_keep_per_group, num_parameters):
                group_sparsity[k] = torch.clamp(1 - param_to_keep / group_max_param, min=0, max=1).item()
                
            return group_sparsity
        
        group_sparsity = compute_the_sparsity_per_group(
            total_parameters_to_keep, 
            group_scores, 
            group_num_parameters, 
            max_sparsity_per_layer=self.max_sparsity_per_layer,
        )
        
        compute_total_keep_parameters = 0
        for k in group_num_parameters:
            compute_total_keep_parameters += (1 - group_sparsity[k]) * group_num_parameters[k]
        
        # for checking
        print(compute_total_keep_parameters, total_parameters_to_keep)
        
        # import pdb; pdb.set_trace()
        
        # print(group_scores)
        # print(group_num_parameters)
        # print(group_sparsity)
        
        layer_sparsity = {
            k: group_sparsity[v]
            for k, v in layer_to_group_mapping.items()
        }
        
        return layer_sparsity

    @print_time
    def compute_importance_scores(self):
        model = self.model
        data_loader = self.data_loader
        loss_func = self.loss_func
        
        names = []
        params = []
        for k, v in model.named_parameters():
            names.append(k)
            params.append(v)
        
        gradients_dict = {k: 0 for k in names}
        
        device = next(iter(model.parameters())).device

        accum_samples = 0
        current_batch_index = 0
        
        for d in data_loader:
            # print(accum_samples)
            if accum_samples >= self.num_samples:
                break
            
            loss, batch_len = loss_func(model, d, device != "cpu")

            accum_samples += batch_len
            current_batch_index += 1

            grads = torch.autograd.grad(loss, params)
            
            assert len(grads) == len(names) == len(params)

            for k, v in zip(names, grads):
                gradients_dict[k] = v.cpu().data.float()

        for k in names:
            # use current_batch_index rather than self.num_samples because sometimes
            # the batch size might not be 1, and the loss is already normalized by 
            # batch size, now when only have to normalize it by num_batches now
            gradients_dict[k] /= current_batch_index
        
        if self.score_compute == "obd":
            # using square of magnitude multiplied by diagonal fisher as importance scores
            importance_measure = {k: (v.cpu().data.float() ** 2) * gradients_dict[k] ** 2 for k, v in self.model.named_parameters()}
        elif self.score_compute == "aobd":
            importance_measure = {k: (v.cpu().data.float().abs()) * gradients_dict[k].abs() for k, v in self.model.named_parameters()}
        elif self.score_compute == "gradient":
            importance_measure = {k: gradients_dict[k].abs() for k, v in self.model.named_parameters()}
        
        return importance_measure