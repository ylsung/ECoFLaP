import torch
import torch.nn as nn
import numpy as np

from time import time
from copy import deepcopy
from functools import partial

from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
from lavis.models.blip2_models.blip2_t5 import Blip2T5
from lavis.models.t5_models.t5 import T5
from lavis.models.clip_models.eva_model import EVA_CLIP
from lavis.compression.pruners.utils import (
    loss_vision_language, loss_language, loss_vision, print_time
)
from lavis.compression.pruners.layer_single_base_pruner import LayerWiseBasePruner, LayerSparsity


def get_module_recursive(base, module_to_process):
    
    if module_to_process == "":
        return base
    
    splits = module_to_process.split(".")
    now = splits.pop(0)
    rest = ".".join(splits)
    
    base = getattr(base, now)

    return get_module_recursive(base, rest)


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


class BLIPT5GlobalPruner(LayerWiseBasePruner):
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
        num_samples=64,
        is_global=False,
        t5_model_prefix="t5_model",
        vit_model_prefix="visual_encoder",
        sparsity_ratio_granularity=None,
        max_sparsity_per_layer=0.8,
        score_method="GradMagSquare_avg",
        num_data_first_stage=128,
        num_noise=1,
        sparsity_dict=None,
        prune_per_model=False,
        iteration=1,
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_loader=data_loader,
            prune_spec=None,
            is_strct_pruning=is_strct_pruning,
            importance_scores_cache=importance_scores_cache,
            keep_indices_or_masks_cache=keep_indices_or_masks_cache,
            is_global=is_global,
            num_samples=num_samples,
            model_prefix="tmp",
            sparsity_ratio_granularity=sparsity_ratio_granularity,
            max_sparsity_per_layer=max_sparsity_per_layer,
            score_method=score_method,
            num_data_first_stage=num_data_first_stage,
            num_noise=num_noise,
            sparsity_dict=sparsity_dict,
        )
        
        self.t5_prune_spec = t5_prune_spec
        self.vit_prune_spec = vit_prune_spec

        self.t5_model_prefix = t5_model_prefix
        self.vit_model_prefix = vit_model_prefix
        
        self.prune_per_model = prune_per_model
        self.iteration = iteration
        
    def compute_importance_scores(self, model, data_loader, loss_func):
        raise NotImplementedError
    
    def get_mask(self, importance_scores, p, max_sparsity_per_layer):
        # Set top (1 - max_sparsity)% of parameters to be very large value to avoid 
        # them being pruned
        
        for k, v in importance_scores.items():
            num_to_set = int(importance_scores[k].numel() * (1 - max_sparsity_per_layer))
            
            if num_to_set > 0:
                threshold, _ = torch.topk(importance_scores[k].flatten(), num_to_set, largest=True)
                threshold = threshold[-1] # take the last value

                importance_scores[k][torch.where(v >= threshold)] = torch.finfo(v.dtype).max
        
        # Flatten all tensors and concatenate them
        all_scores = torch.cat([t.flatten() for t in importance_scores.values()])
        
        # Sort and find the threshold
        num_to_zero_out = int(p * all_scores.numel())
        threshold, _ = torch.topk(all_scores, num_to_zero_out, largest=False)
        threshold = threshold[-1]
        
        # Create mask based on threshold
        masks = {}
        for k, v in importance_scores.items():
            masks[k] = (v > threshold).type(v.dtype)
        
        return masks
    
    def get_layerwise_mask(self, importance_scores, p):
        # Set top (1 - max_sparsity)% of parameters to be very large value to avoid 
        # them being pruned
        
        masks = {}
        for k, v in importance_scores.items():
            all_scores = importance_scores[k].flatten()
            num_to_zero_out = int(p * all_scores.numel())
            threshold, _ = torch.topk(all_scores, num_to_zero_out, largest=False)
            threshold = threshold[-1]

            masks[k] = (v > threshold).type(v.dtype)

        return masks
    
    def forward_to_cache(self, model, batch, device):
        return model(batch)
    
    def global_iterative_pruning(self, target_sparsity, dict_layers_to_prune, iteratation=1, max_sparsity_per_layer=1.0):
        
        masks = None
        for i in range(1, iteratation+1):
            p_i = target_sparsity ** (iteratation / i) # Compute modified sparsity for the i^th iteration
            
            importance_measure = self.compute_importance_scores(
                self.model, self.data_loader, dict_layers_to_prune, loss_vision_language
            )
            
            importance_measure = {k: v for k, v in importance_measure.items() if k in dict_layers_to_prune}
            
            if masks is not None:
                # Apply mask to importance scores (this step is to simulate pruning in iterations)
                for k in importance_measure:
                    importance_measure[k] *= masks[k]

            if self.is_global and not self.prune_per_model:
                # total global
                print("global")
                masks = self.get_mask(importance_measure, p_i, max_sparsity_per_layer)
            elif self.is_global and self.prune_per_model:
                print("model-level global")
                vision_scores = {k: v for k, v in importance_measure.items() if k.startswith(self.vit_model_prefix)}
                language_scores = {k: v for k, v in importance_measure.items() if k.startswith(self.t5_model_prefix)}
                vision_masks = self.get_mask(vision_scores, p_i, max_sparsity_per_layer)
                language_masks = self.get_mask(language_scores, p_i, max_sparsity_per_layer)
                
                vision_masks.update(language_masks)
                
                masks = vision_masks
            else:
                print("layer-wise")
                masks = self.get_layerwise_mask(importance_measure, p_i)
            
            # prune the model
            for k, v in self.model.named_parameters():
                if k in masks:
                    v.data *= masks[k].type(v.dtype).to(v.device)
                    
            print(f"Step {i}, target sparsity: {p_i:.4f}")
            
        for k, v in self.model.named_parameters():
            print(k, " sparsity: ", (v == 0).float().sum() / v.numel())
        
        return self.model

    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):
        print("In: ", self.pruner_name)
        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)
        
        if self.t5_prune_spec is None or self.vit_prune_spec is None:
            return self.model, None
        
        _, vit_keep_ratio, _, _ = self.convert_spec_to_list(self.vit_prune_spec)
        _, t5_keep_ratio, _, _ = self.convert_spec_to_list(self.t5_prune_spec) 
        assert vit_keep_ratio == t5_keep_ratio
        
        def check(name, v):
            if len(v.shape) == 2 and \
                    ".block" in name and \
                    "relative_attention_bias.weight" not in name and \
                    (name.startswith(self.t5_model_prefix) or \
                        name.startswith(self.vit_model_prefix)):
                return True
            return False

        parameters_to_prune = {
            k: v for k, v in self.model.named_parameters() if check(k, v)
        }

        self.model = self.global_iterative_pruning(
            1 - vit_keep_ratio,
            parameters_to_prune, 
            iteratation=self.iteration, 
            max_sparsity_per_layer=1.0,
        )

        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, None
    
@registry.register_pruner("blipt5_global_mag_pruner")
class BLIPT5GlobalMagPruner(BLIPT5GlobalPruner):
    pruner_name = "blipt5_global_mag_pruner"
    
    def compute_importance_scores(self, model, data_loader=None, dict_layers_to_prune={}, loss_func=None):
        return {k: v.data.float().cpu() for k, v in model.named_parameters()}
    

@registry.register_pruner("blipt5_global_gradmagabs_pruner")
class BLIPT5GlobalGradMagAbsPruner(BLIPT5GlobalPruner):
    pruner_name = "blipt5_global_gradmagabs_pruner"
    
    @print_time
    def compute_importance_scores(self, model, data_loader=None, dict_layers_to_prune={}, loss_func=None):
        
        names = []
        params = []
        for k, v in model.named_parameters():
            if k in dict_layers_to_prune:
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
                gradients_dict[k] += v.cpu().data.float().abs()

        for k in names:
            # use current_batch_index rather than self.num_samples because sometimes
            # the batch size might not be 1, and the loss is already normalized by 
            # batch size, now when only have to normalize it by num_batches now
            gradients_dict[k] /= current_batch_index

        importance_measure = {k: (v.cpu().data.float().abs()) * gradients_dict[k].abs() for k, v in zip(names, params)}
        
        return importance_measure
    

@registry.register_pruner("blipt5_global_mezo_pruner")
class BLIPT5GlobalMeZoPruner(BLIPT5GlobalPruner):
    pruner_name = "blipt5_global_mezo_pruner"
    
    def zo_perturb_parameters(self, params, random_seed=1, scaling_factor=1, zo_eps=1e-3):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed)
        
        for param in params:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * zo_eps
    
    @print_time
    def compute_importance_scores(self, model, data_loader=None, dict_layers_to_prune={}, loss_func=None):
        
        names = []
        params = []
        model.eval()
        for k, v in model.named_parameters():  
            if k in dict_layers_to_prune:
                names.append(k)
                params.append(v)
        
        gradients_dict = {k: 0 for k in names}
        
        device = next(iter(model.parameters())).device

        accum_samples = 0
        current_batch_index = 0
        
        zo_eps = 1e-3
        
        n_mezo = self.num_noise
        
        for i, (name, param) in enumerate(zip(names, params)):
            print(i, name)
            accum_samples = 0
            current_batch_index = 0
            
            for d in data_loader:
                if accum_samples >= self.num_samples:
                    break
                
                per_gradients_dict = {name: 0}
                
                for _ in range(n_mezo):
                    
                    if accum_samples >= self.num_samples:
                        break
                    
                    zo_random_seed = np.random.randint(1000000000)
                    
                    self.zo_perturb_parameters([param], random_seed=zo_random_seed, scaling_factor=1, zo_eps=zo_eps)
                    with torch.no_grad():
                        loss1, batch_len = loss_func(model, d, device != "cpu")
                    
                    self.zo_perturb_parameters([param], random_seed=zo_random_seed, scaling_factor=-2, zo_eps=zo_eps)
                    with torch.no_grad():
                        loss2, batch_len = loss_func(model, d, device != "cpu")
                
                    # recover the weight
                    self.zo_perturb_parameters([param], random_seed=zo_random_seed, scaling_factor=1, zo_eps=zo_eps)

                    accum_samples += batch_len
                    current_batch_index += 1
                    
                    projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()
                    
                    # print(zo_random_seed, loss1, loss2, projected_grad)

                    # gradients_dict = self.zo_gradients(gradients_dict, names, params, projected_grad, random_seed=zo_random_seed)
                    
                    torch.manual_seed(zo_random_seed)
                    per_gradients_dict[name] += abs(projected_grad)
                        
                gradients_dict[name] += torch.FloatTensor([per_gradients_dict[name]]).abs()
                
        importance_measure = {k: gradients_dict[k].abs() for k, v in zip(names, params)}
            
        return importance_measure