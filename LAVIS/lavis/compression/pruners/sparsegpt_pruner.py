import torch
import math
import torch.nn as nn

from time import time
from copy import deepcopy
from functools import partial
import transformers

from lavis.common.registry import registry
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

class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)
        
        if (torch.isinf(H) * (H > 0)).float().sum() > 0:
            # positive inf value
            pos = torch.isinf(H) * (H > 0)
            H[pos] = torch.quantile(H, 0.999)
            
        if (torch.isinf(H) * (H < 0)).float().sum() > 0:
            # negative inf value
            pos = torch.isinf(H) * (H < 0)
            H[pos] = torch.quantile(H, 0.001)
            
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        
        while True:
            try:
                decompose_H = torch.linalg.cholesky(H)
                
                if not torch.isnan(decompose_H).any():
                    H = decompose_H
                    break
                
                if torch.isinf(damp).any():
                    import pdb; pdb.set_trace()
                # not a positive semi-definite matrix
                H[diag, diag] += damp
            except:
                # not a positive semi-definite matrix
                H[diag, diag] += damp
        # H[diag, diag] += damp
        # H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        
        if (torch.isinf(H) * (H > 0)).float().sum() > 0:
            # positive inf value
            pos = torch.isinf(H) * (H > 0)
            H[pos] = torch.quantile(H, 0.999)
            
        if (torch.isinf(H) * (H < 0)).float().sum() > 0:
            # negative inf value
            pos = torch.isinf(H) * (H < 0)
            H[pos] = torch.quantile(H, 0.001)
            
        damp = percdamp * torch.mean(torch.diag(H).abs())
        diag = torch.arange(self.columns, device=self.dev)
        
        while True:
            try:
                decompose_H = torch.linalg.cholesky(H, upper=True)
                
                if not torch.isnan(decompose_H).any():
                    H = decompose_H
                    break
                # not a positive semi-definite matrix
                H[diag, diag] += damp
            except:
                # not a positive semi-definite matrix
                H[diag, diag] += damp

        # H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        
        
        s = W ** 2 / (torch.diag(Hinv).reshape((1, -1))) ** 2
        
        setattr(self.layer.weight, "importance_score", s.cpu().abs().mean().item())

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


@registry.register_pruner("t5_sparsegpt_pruner")
class T5LayerSparseGPTPruner(LayerWiseBasePruner):
    pruner_name = "t5_sparsegpt_pruner"
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
        score_method="obd_avg",
        num_data_first_stage=128,
        num_noise=1,
        sparsity_dict=None,
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
            sparsity_ratio_granularity=sparsity_ratio_granularity,
            max_sparsity_per_layer=max_sparsity_per_layer,
            score_method=score_method,
            num_data_first_stage=num_data_first_stage,
            num_noise=num_noise,
            sparsity_dict=sparsity_dict,
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
    
    def check_sparsity(self, model, module_to_process="encoder.block"):
        use_cache = getattr(model, self.model_prefix).config.use_cache 
        getattr(model, self.model_prefix).config.use_cache = False 

        layers = get_module_recursive(model, module_to_process)
        count = 0 
        total_params = 0
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                count += (W==0).sum().item()
                total_params += W.numel()

                sub_count += (W==0).sum().item()
                sub_params += W.numel()

            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

        getattr(model, self.model_prefix).config.use_cache = use_cache 
        return float(count)/total_params 
    
    def forward_to_cache(self, model, batch):
        return model(batch)
    
    def prepare_calibration_input_encoder(self, model, dataloader, device, model_prefix, n_samples, module_to_process="encoder.block"):
        use_cache = getattr(model, model_prefix).config.use_cache
        getattr(model, model_prefix).config.use_cache = False
        # layers = model.encoder.block
        layers = get_module_recursive(model, module_to_process)

        dtype = next(iter(model.parameters())).dtype
        # inps = torch.zeros((2, max_txt_len, getattr(model, self.model_prefix).config.d_model), dtype=dtype, device=device)
        inps = []
        # caches = {'i': 0}
        
        caches = []
        
        keys_to_cache = [
            "attention_mask", "position_bias", "encoder_attention_mask", "encoder_decoder_position_bias",
            "layer_head_mask", "cross_attn_layer_head_mask", "encoder_hidden_states",
        ]
        
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, inp, **kwargs):
                # length = inp.shape[1]
                # inps[cache['i'], :length] = inp
                inps.append(inp)
                inps[-1].requires_grad = False
                # cache['i'] += 1
                
                cache = {}
                for k in keys_to_cache:
                    cache[k] = kwargs[k]
                caches.append(cache)
                # cache['attention_mask'] = kwargs['attention_mask']
                # cache['position_bias'] = kwargs['position_bias']
                raise ValueError

        layers[0] = Catcher(layers[0])
        for i, batch in enumerate(dataloader):
            if i >= n_samples:
                break
            try:
                # batch = process_input(batch, t5_tokenizer)
                # print(f"In {i}: ", (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000)
                # model(batch)
                self.forward_to_cache(model, batch)
            except ValueError:
                pass 
        layers[0] = layers[0].module
        
        # outs = torch.zeros_like(inps)
        outs = [None] * len(inps)
        # attention_mask = cache['attention_mask']
        # position_bias = cache['position_bias']

        getattr(model, model_prefix).config.use_cache = use_cache
        
        # del caches["i"]
        
        # print(caches)

        return inps, outs, caches
    
    @print_time
    def _prune(self, model, dataloader, device, model_prefix, module_to_process="encoder.block", n_samples=64, sparsity_ratio=0.5):
        use_cache = getattr(model, model_prefix).config.use_cache 
        getattr(model, model_prefix).config.use_cache = False 

        print("loading calibdation data")
        with torch.no_grad():
            inps, outs, caches = self.prepare_calibration_input_encoder(model, dataloader, device, model_prefix, n_samples, module_to_process)

        n_samples = min(n_samples, len(inps))

        layers = get_module_recursive(model, module_to_process)
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            #     dev = model.hf_device_map[f"model.layers.{i}"]
            #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = SparseGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(n_samples):
                with torch.no_grad():
                    with model.maybe_autocast(dtype=torch.bfloat16):
                        outs[j] = layer(inps[j], **caches[j])[0]
            for h in handles:
                h.remove()

            for name in subset:
                assert wrapped_layers[name].nsamples == len(inps)
                print(f"pruning layer {i} name {name}")
                
                sparsity_key = f"{module_to_process}.{i}.{name}.weight"
                wrapped_layers[name].fasterprune(sparsity_ratio[sparsity_key], prune_n=self.prune_n, prune_m=self.prune_m, percdamp=0.01, blocksize=128)
                # print(
                #     sparsity_ratio[sparsity_key], 
                #     (wrapped_layers[name].layer.weight != 0).float().sum() / wrapped_layers[name].layer.weight.numel
                # )
                wrapped_layers[name].free()

            for j in range(n_samples):
                with torch.no_grad():
                    with model.maybe_autocast(dtype=torch.bfloat16):
                        outs[j] = layer(inps[j], **caches[j])[0]
            inps, outs = outs, inps

        getattr(model, model_prefix).config.use_cache = use_cache 
        torch.cuda.empty_cache()
        
        return model
    
    def get_sparsity(self, original_sparsity, sparsity_ratio_granularity=None):
        if self.sparsity_dict is not None:
            import yaml
            with open(self.sparsity_dict, "r") as f:
                return yaml.load(f, Loader=yaml.FullLoader)

        if sparsity_ratio_granularity == None:
            layer_to_group_mapping = {}
        
        else:
            def check(name, v):
                if len(v.shape) == 2 and \
                        ".block" in name and \
                            "relative_attention_bias.weight" not in name and \
                                name.startswith(self.model_prefix):
                    return True
                return False
            parameters_to_prune = [
                k for k, v in self.model.named_parameters() if check(k, v)
            ]

            if sparsity_ratio_granularity == "layer":
                layer_to_group_mapping = {
                    k: k
                    for k in parameters_to_prune
                }
            elif sparsity_ratio_granularity == "block":
                layer_to_group_mapping = {
                    k: ".".join(k.split(".")[:4])
                    for k in parameters_to_prune
                }
            else:
                raise NotImplementedError
        
        sparsity_module = LayerSparsity(
            self.model, 
            self.data_loader, 
            loss_language, 
            self.num_data_first_stage,
            original_sparsity,
            self.max_sparsity_per_layer,
            self.score_method,
            self.num_noise,
            layer_to_group_mapping,
        )
        
        return sparsity_module.return_sparsity()
        
    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):
        print("In: ", self.pruner_name)
        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)

        if self.prune_spec is None:
            return self.model, None

        _, keep_ratio, _, _ = self.convert_spec_to_list(self.prune_spec)
        
        sparsity_ratio = 1 - keep_ratio
        
        sparsity_dict = self.get_sparsity(
            sparsity_ratio,
            sparsity_ratio_granularity=self.sparsity_ratio_granularity
        )
        
        self.model = self._prune(
            self.model, self.data_loader, device, 
            model_prefix=self.model_prefix,
            module_to_process=f"{self.model_prefix}.encoder.block",
            n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
        )
        self.model = self._prune(
            self.model, self.data_loader, device, 
            model_prefix=self.model_prefix,
            module_to_process=f"{self.model_prefix}.decoder.block",
            n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
        )

        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, sparsity_dict


@registry.register_pruner("vit_sparsegpt_pruner")
class VITLayerSparseGPTPruner(LayerWiseBasePruner):
    pruner_name = "vit_sparsegpt_pruner"
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
        model_prefix="visual",
        sparsity_ratio_granularity=None,
        max_sparsity_per_layer=0.8,
        score_method="obd_avg",
        num_data_first_stage=128,
        num_noise=1,
        sparsity_dict=None,
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
            sparsity_ratio_granularity=sparsity_ratio_granularity,
            max_sparsity_per_layer=max_sparsity_per_layer,
            score_method=score_method,
            num_data_first_stage=num_data_first_stage,
            num_noise=num_noise,
            sparsity_dict=sparsity_dict,
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
    
    def check_sparsity(self, model, module_to_process="encoder.block"):
        layers = get_module_recursive(model, module_to_process)
        count = 0 
        total_params = 0
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                count += (W==0).sum().item()
                total_params += W.numel()

                sub_count += (W==0).sum().item()
                sub_params += W.numel()

            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

        return float(count)/total_params 
    
    def forward_to_cache(self, model, batch):
        return model.encode_image(batch["image"])
    
    def prepare_calibration_input_encoder(self, model, dataloader, device, model_prefix, n_samples, module_to_process="encoder.block"):
        layers = get_module_recursive(model, module_to_process)

        dtype = next(iter(model.parameters())).dtype
        # inps = torch.zeros((2, max_txt_len, getattr(model, self.model_prefix).config.d_model), dtype=dtype, device=device)
        inps = []
        # caches = {'i': 0}
        
        print(dtype)
        
        caches = []
        
        keys_to_cache = [
            "rel_pos_bias"
        ]

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, inp, rel_pos_bias):
                inps.append(inp)
                inps[-1].requires_grad = False
                
                cache = {}
                cache["rel_pos_bias"] = rel_pos_bias
                caches.append(cache)
                raise ValueError

        layers[0] = Catcher(layers[0])
        
        for i, batch in enumerate(dataloader):
            if i >= n_samples:
                break
            try:
                # model(batch)
                # import pdb; pdb.set_trace()
                # for k, v in batch.items():
                #     if isinstance(v, torch.FloatTensor):
                #         batch[k] = v.type(dtype)
                        
                #         print(k, batch[k].type)
                # model.encode_image(batch["image"])
                self.forward_to_cache(model, batch)
            except ValueError:
                pass 
        layers[0] = layers[0].module

        outs = [None] * len(inps)

        return inps, outs, caches
    
    @print_time
    def _prune(self, model, dataloader, device, model_prefix, module_to_process="encoder.block", n_samples=64, sparsity_ratio=0.5):
        print("loading calibdation data")
        with torch.no_grad():
            inps, outs, caches = self.prepare_calibration_input_encoder(model, dataloader, device, model_prefix, n_samples, module_to_process)

        n_samples = min(n_samples, len(inps))

        layers = get_module_recursive(model, module_to_process)
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            #     dev = model.hf_device_map[f"model.layers.{i}"]
            #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = SparseGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    # print(inp[0].data.shape)
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(n_samples):
                with torch.no_grad():
                    with model.maybe_autocast():
                        outs[j] = layer(inps[j], **caches[j])

            for h in handles:
                h.remove()

            for name in subset:
                assert wrapped_layers[name].nsamples == len(inps)
                print(f"pruning layer {i} name {name}")

                sparsity_key = f"{module_to_process}.{i}.{name}.weight"
                wrapped_layers[name].fasterprune(sparsity_ratio[sparsity_key], prune_n=self.prune_n, prune_m=self.prune_m, percdamp=0.01, blocksize=128)
                wrapped_layers[name].free()

            for j in range(n_samples):
                with torch.no_grad():
                    with model.maybe_autocast():
                        outs[j] = layer(inps[j], **caches[j])
            inps, outs = outs, inps

        torch.cuda.empty_cache()

        return model
    
    def get_sparsity(self, original_sparsity, sparsity_ratio_granularity=None):
        if self.sparsity_dict is not None:
            import yaml
            with open(self.sparsity_dict, "r") as f:
                sparsity_dict = yaml.load(f, Loader=yaml.FullLoader)
                
            sparsity_dict = {k.replace("visual_encoder.", "visual."): v for k, v in sparsity_dict.items()}
            
            if "visual.blocks.39.attn.qkv.weight" not in sparsity_dict:
                # get from multi-modal pruning
                sparsity_dict["visual.blocks.39.attn.qkv.weight"] = 0
                sparsity_dict["visual.blocks.39.attn.proj.weight"] = 0
                sparsity_dict["visual.blocks.39.mlp.fc1.weight"] = 0
                sparsity_dict["visual.blocks.39.mlp.fc2.weight"] = 0
            
            return sparsity_dict

        if sparsity_ratio_granularity == None:
            layer_to_group_mapping = {}
        
        else:
            def check(name, v):
                if len(v.shape) == 2 and \
                        ".blocks" in name and \
                            name.startswith(self.model_prefix):
                    return True
                return False
            parameters_to_prune = [
                k for k, v in self.model.named_parameters() if check(k, v)
            ]

            if sparsity_ratio_granularity == "layer":
                layer_to_group_mapping = {
                    k: k
                    for k in parameters_to_prune
                }
            elif sparsity_ratio_granularity == "block":
                layer_to_group_mapping = {
                    k: ".".join(k.split(".")[:3])
                    for k in parameters_to_prune
                }
            else:
                raise NotImplementedError
        
        sparsity_module = LayerSparsity(
            self.model, 
            self.data_loader, 
            loss_vision, 
            self.num_data_first_stage,
            original_sparsity,
            self.max_sparsity_per_layer,
            self.score_method,
            self.num_noise,
            layer_to_group_mapping,
        )
        
        return sparsity_module.return_sparsity()

    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):
        print("In: ", self.pruner_name)
        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)

        if self.prune_spec is None:
            return self.model, None

        _, keep_ratio, _, _ = self.convert_spec_to_list(self.prune_spec)
        
        sparsity_ratio = 1 - keep_ratio
        
        sparsity_dict = self.get_sparsity(
            sparsity_ratio,
            sparsity_ratio_granularity=self.sparsity_ratio_granularity
        )
        
        self.model = self._prune(
            self.model, self.data_loader, device, 
            model_prefix=self.model_prefix,
            module_to_process=f"{self.model_prefix}.blocks",
            n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
        )

        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, sparsity_dict


@registry.register_pruner("blipt5_sparsegpt_pruner")
class BLIPT5LayerSparseGPTPruner(LayerWiseBasePruner):
    pruner_name = "blipt5_sparsegpt_pruner"
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
        score_method="obd_avg",
        num_data_first_stage=128,
        num_noise=1,
        sparsity_dict=None,
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
        
        assert t5_pruning_method is not None
        assert vit_pruning_method is not None
        
        self.t5_model_prefix = t5_model_prefix
        self.vit_model_prefix = vit_model_prefix
        
        # from lavis.compression import load_pruner

        # t5_config = {
        #     "prune_spec": t5_prune_spec,
        #     "importance_scores_cache": t5_importance_scores_cache,
        #     "keep_indices_cache": t5_keep_indices_or_masks_cache,
        #     "is_strct_pruning": is_strct_pruning,
        #     "is_global": is_global,
        #     "model_prefix": self.t5_model_prefix,
        # }
        # self.t5_pruner = load_pruner(t5_pruning_method, model, data_loader, cfg=t5_config)

        # vit_config = {
        #     "prune_spec": vit_prune_spec,
        #     "importance_scores_cache": vit_importance_scores_cache,
        #     "keep_indices_cache": vit_keep_indices_or_masks_cache,
        #     "is_strct_pruning": is_strct_pruning,
        #     "is_global": is_global,
        #     "model_prefix": self.vit_model_prefix,
        # }
        # self.vit_pruner = load_pruner(vit_pruning_method, model, data_loader, cfg=vit_config)
        
        # self.loss_func = loss_vision_language
        
    def get_sparsity(self, original_sparsity, sparsity_ratio_granularity=None):
        if self.sparsity_dict is not None:
            import yaml
            with open(self.sparsity_dict, "r") as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        
        if sparsity_ratio_granularity == None:
            layer_to_group_mapping = {}
        
        else:
            def check(name, v):
                if len(v.shape) == 2 and \
                     ".block" in name and \
                        "relative_attention_bias.weight" not in name and \
                        (name.startswith(self.t5_model_prefix) or \
                            name.startswith(self.vit_model_prefix)):
                    return True
                return False
            parameters_to_prune = [
                k for k, v in self.model.named_parameters() if check(k, v)
            ]

            if sparsity_ratio_granularity == "model":
                
                def return_group(name):
                    if name.startswith(self.t5_model_prefix):
                        return self.t5_model_prefix
                    elif name.startswith(self.vit_model_prefix):
                        return self.vit_model_prefix
                    else:
                        return "other"
                
                layer_to_group_mapping = {
                    k: return_group(k)
                    for k in parameters_to_prune
                }
                
            elif sparsity_ratio_granularity == "layer":
                layer_to_group_mapping = {
                    k: k
                    for k in parameters_to_prune
                }
            elif sparsity_ratio_granularity == "block":
                def return_group(name):
                    if name.startswith(self.t5_model_prefix):
                        return ".".join(name.split(".")[:4])
                    elif name.startswith(self.vit_model_prefix):
                        return ".".join(name.split(".")[:3])
                    else:
                        return "other"
                layer_to_group_mapping = {
                    k: return_group(k)
                    for k in parameters_to_prune
                }
            else:
                raise NotImplementedError
        
        sparsity_module = LayerSparsity(
            self.model, 
            self.data_loader, 
            loss_vision_language, 
            self.num_data_first_stage,
            original_sparsity,
            self.max_sparsity_per_layer,
            self.score_method,
            self.num_noise,
            layer_to_group_mapping,
        )
        
        return sparsity_module.return_sparsity()
        
    def forward_to_cache(self, model, batch):
        return model(batch)
        
    # def t5_prepare_calibration_input_encoder(self, model, dataloader, device, model_prefix, n_samples, module_to_process="encoder.block"):
    #     use_cache = getattr(model, model_prefix).config.use_cache
    #     getattr(model, model_prefix).config.use_cache = False
    #     # layers = model.encoder.block
    #     layers = get_module_recursive(model, module_to_process)

    #     dtype = next(iter(model.parameters())).dtype
    #     # inps = torch.zeros((2, max_txt_len, getattr(model, self.model_prefix).config.d_model), dtype=dtype, device=device)
    #     inps = []

    #     caches = []
        
    #     keys_to_cache = [
    #         "attention_mask", "position_bias", "encoder_attention_mask", "encoder_decoder_position_bias",
    #         "layer_head_mask", "cross_attn_layer_head_mask", "encoder_hidden_states",
    #     ]
        
    #     class Catcher(nn.Module):
    #         def __init__(self, module):
    #             super().__init__()
    #             self.module = module
    #         def forward(self, inp, **kwargs):
    #             # length = inp.shape[1]
    #             # inps[cache['i'], :length] = inp
    #             inps.append(inp)
    #             inps[-1].requires_grad = False
    #             # cache['i'] += 1
                
    #             cache = {}
    #             for k in keys_to_cache:
    #                 cache[k] = kwargs[k]
    #             caches.append(cache)
    #             # cache['attention_mask'] = kwargs['attention_mask']
    #             # cache['position_bias'] = kwargs['position_bias']
    #             raise ValueError

    #     layers[0] = Catcher(layers[0])
    #     for i, batch in enumerate(dataloader):
    #         if i >= n_samples:
    #             break
    #         try:
    #             # batch = process_input(batch, t5_tokenizer)
    #             print(f"In {i}: ", (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000)
    #             # model(batch)
    #             self.forward_to_cache(model, batch)
    #         except ValueError:
    #             pass 
    #     layers[0] = layers[0].module

    #     # outs = torch.zeros_like(inps)
    #     outs = [None] * len(inps)
    #     # attention_mask = cache['attention_mask']
    #     # position_bias = cache['position_bias']

    #     getattr(model, model_prefix).config.use_cache = use_cache
        
    #     # del caches["i"]
        
    #     # print(caches)

    #     return inps, outs, caches
    
    # @print_time
    # def _t5_prune(self, model, dataloader, device, model_prefix, module_to_process="encoder.block", n_samples=64, sparsity_ratio=0.5):
    #     use_cache = getattr(model, model_prefix).config.use_cache 
    #     getattr(model, model_prefix).config.use_cache = False 

    #     print("loading calibdation data")
    #     with torch.no_grad():
    #         inps, outs, caches = self.t5_prepare_calibration_input_encoder(model, dataloader, device, model_prefix, n_samples, module_to_process)

    #     n_samples = min(n_samples, len(inps))

    #     layers = get_module_recursive(model, module_to_process)
    #     for i in range(len(layers)):
    #         layer = layers[i]
    #         subset = find_layers(layer)

    #         # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
    #         #     dev = model.hf_device_map[f"model.layers.{i}"]
    #         #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

    #         wrapped_layers = {}
    #         for name in subset:
    #             wrapped_layers[name] = SparseGPT(subset[name])

    #         def add_batch(name):
    #             def tmp(_, inp, out):
    #                 wrapped_layers[name].add_batch(inp[0].data, out.data)
    #             return tmp

    #         handles = []
    #         for name in wrapped_layers:
    #             handles.append(subset[name].register_forward_hook(add_batch(name)))

    #         for j in range(n_samples):
    #             with torch.no_grad():
    #                 outs[j] = layer(inps[j], **caches[j])[0]
    #         for h in handles:
    #             h.remove()

    #         for name in subset:
    #             print(f"pruning layer {i} name {name}")
    #             W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

    #             W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
    #             if self.prune_n != 0:
    #                 # structured n:m sparsity
    #                 for ii in range(W_metric.shape[1]):
    #                     if ii % self.prune_m == 0:
    #                         tmp = W_metric[:,ii:(ii+self.prune_m)].float()
    #                         W_mask.scatter_(1,ii+torch.topk(tmp, self.prune_n, dim=1, largest=False)[1], True)
    #             else:
    #                 sort_res = torch.sort(W_metric, dim=-1, stable=True)

    #                 # unstructured pruning
    #                 indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
    #                 W_mask.scatter_(1, indices, True)

    #             subset[name].weight.data[W_mask] = 0  ## set weights to zero 

    #         for j in range(n_samples):
    #             with torch.no_grad():
    #                 outs[j] = layer(inps[j], **caches[j])[0]
    #         inps, outs = outs, inps

    #     getattr(model, model_prefix).config.use_cache = use_cache 
    #     torch.cuda.empty_cache()
        
    #     return model

    # def vit_prepare_calibration_input_encoder(self, model, dataloader, device, model_prefix, n_samples, module_to_process="encoder.block"):
    #     layers = get_module_recursive(model, module_to_process)

    #     dtype = next(iter(model.parameters())).dtype
    #     # inps = torch.zeros((2, max_txt_len, getattr(model, self.model_prefix).config.d_model), dtype=dtype, device=device)
    #     inps = []
    #     # caches = {'i': 0}
        
    #     caches = []
        
    #     keys_to_cache = [
    #         "rel_pos_bias"
    #     ]
        
    #     class Catcher(nn.Module):
    #         def __init__(self, module):
    #             super().__init__()
    #             self.module = module
    #         def forward(self, inp, rel_pos_bias):
    #             inps.append(inp)
    #             inps[-1].requires_grad = False
                
    #             cache = {}
    #             cache["rel_pos_bias"] = rel_pos_bias
    #             caches.append(cache)
    #             raise ValueError

    #     layers[0] = Catcher(layers[0])
    #     for i, batch in enumerate(dataloader):
    #         if i >= n_samples:
    #             break
    #         try:
    #             # batch = process_input(batch, t5_tokenizer)
    #             print(f"In {i}: ", (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000)
    #             self.forward_to_cache(model, batch)
    #         except ValueError:
    #             pass 

    #     layers[0] = layers[0].module

    #     outs = [None] * len(inps)

    #     return inps, outs, caches
    
    # @print_time
    # def _vit_prune(self, model, dataloader, device, model_prefix, module_to_process="encoder.block", n_samples=64, sparsity_ratio=0.5):
    #     print("loading calibdation data")
    #     with torch.no_grad():
    #         inps, outs, caches = self.vit_prepare_calibration_input_encoder(model, dataloader, device, model_prefix, n_samples, module_to_process)

    #     n_samples = min(n_samples, len(inps))

    #     layers = get_module_recursive(model, module_to_process)
    #     for i in range(len(layers)):
    #         layer = layers[i]
    #         subset = find_layers(layer)

    #         # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
    #         #     dev = model.hf_device_map[f"model.layers.{i}"]
    #         #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

    #         wrapped_layers = {}
    #         for name in subset:
    #             wrapped_layers[name] = SparseGPT(subset[name])

    #         def add_batch(name):
    #             def tmp(_, inp, out):
    #                 # print(inp[0].data.shape)
    #                 wrapped_layers[name].add_batch(inp[0].data, out.data)
    #             return tmp

    #         handles = []
    #         for name in wrapped_layers:
    #             handles.append(subset[name].register_forward_hook(add_batch(name)))

    #         for j in range(n_samples):
    #             with torch.no_grad():
    #                 with model.maybe_autocast():
    #                 # import pdb; pdb.set_trace()
    #                     outs[j] = layer(inps[j], **caches[j])
    #         for h in handles:
    #             h.remove()

    #         for name in subset:
    #             print(f"pruning layer {i} name {name}")
    #             W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

    #             W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
    #             if self.prune_n != 0:
    #                 # structured n:m sparsity
    #                 for ii in range(W_metric.shape[1]):
    #                     if ii % self.prune_m == 0:
    #                         tmp = W_metric[:,ii:(ii+self.prune_m)].float()
    #                         W_mask.scatter_(1,ii+torch.topk(tmp, self.prune_n, dim=1, largest=False)[1], True)
    #             else:
    #                 sort_res = torch.sort(W_metric, dim=-1, stable=True)

    #                 # unstructured pruning
    #                 indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
    #                 W_mask.scatter_(1, indices, True)

    #             subset[name].weight.data[W_mask] = 0  ## set weights to zero 

    #         for j in range(n_samples):
    #             with torch.no_grad():
    #                 with model.maybe_autocast():
    #                     outs[j] = layer(inps[j], **caches[j])
    #         inps, outs = outs, inps

    #     torch.cuda.empty_cache()

    #     return model

    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):
        print("In: ", self.pruner_name)
        dtype_record, requires_grad_record, device = self.model_setup_and_record_attributes(self.model)

        global_sparsity_dict = None
        if self.sparsity_ratio_granularity is not None: 
            _, vit_keep_ratio, _, _ = self.convert_spec_to_list(self.vit_prune_spec)
            _, t5_keep_ratio, _, _ = self.convert_spec_to_list(self.t5_prune_spec) 
            assert vit_keep_ratio == t5_keep_ratio

            global_sparsity_dict = self.get_sparsity(
                1 - vit_keep_ratio, # same as 1 - t5_keep_ratio
                sparsity_ratio_granularity=self.sparsity_ratio_granularity
            )
            
        if self.vit_prune_spec is not None:
            _, keep_ratio, _, _ = self.convert_spec_to_list(self.vit_prune_spec)
        
            sparsity_ratio = 1 - keep_ratio
            
            if global_sparsity_dict is not None:
                sparsity_dict = global_sparsity_dict
            else:
                sparsity_dict = self.get_sparsity(
                    sparsity_ratio,
                    sparsity_ratio_granularity=None
                )
            
            # self.model = self._vit_prune(
            #     self.model, self.data_loader, device, 
            #     model_prefix=self.vit_model_prefix,
            #     module_to_process=f"{self.vit_model_prefix}.blocks",
            #     n_samples=self.num_samples, sparsity_ratio=sparsity_ratio,
            # )
            
            _vit_prune = partial(VITLayerSparseGPTPruner._prune, self)
            self.prepare_calibration_input_encoder = partial(
                VITLayerSparseGPTPruner.prepare_calibration_input_encoder,
                self,
                )
            
            self.model = _vit_prune(
                self.model, self.data_loader, device, 
                model_prefix=self.vit_model_prefix,
                module_to_process=f"{self.vit_model_prefix}.blocks",
                n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
            )
            
        if self.t5_prune_spec is not None:
            _, keep_ratio, _, _ = self.convert_spec_to_list(self.t5_prune_spec)
        
            sparsity_ratio = 1 - keep_ratio
            
            if global_sparsity_dict is not None:
                sparsity_dict = global_sparsity_dict
            else:
                sparsity_dict = self.get_sparsity(
                    sparsity_ratio,
                    sparsity_ratio_granularity=None
                )
            
            _t5_prune = partial(T5LayerSparseGPTPruner._prune, self)
            self.prepare_calibration_input_encoder = partial(
                T5LayerSparseGPTPruner.prepare_calibration_input_encoder,
                self,
                )
            
            self.model = _t5_prune(
                self.model, self.data_loader, device, 
                model_prefix=self.t5_model_prefix,
                module_to_process=f"{self.t5_model_prefix}.encoder.block",
                n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
            )
            
            self.model = _t5_prune(
                self.model, self.data_loader, device, 
                model_prefix=self.t5_model_prefix,
                module_to_process=f"{self.t5_model_prefix}.decoder.block",
                n_samples=self.num_samples, sparsity_ratio=sparsity_dict,
            )

            # self.model = self._t5_prune(
            #     self.model, self.data_loader, device, 
            #     model_prefix=self.t5_model_prefix,
            #     module_to_process=f"{self.t5_model_prefix}.encoder.block",
            #     n_samples=self.num_samples, sparsity_ratio=sparsity_ratio,
            # )
            # self.model = self._t5_prune(
            #     self.model, self.data_loader, device, 
            #     model_prefix=self.t5_model_prefix,
            #     module_to_process=f"{self.t5_model_prefix}.decoder.block",
            #     n_samples=self.num_samples, sparsity_ratio=sparsity_ratio,
            # )

        # let the pruned model has the original
        self.model_reset(self.model, dtype_record, requires_grad_record, device)
        
        return self.model, global_sparsity_dict