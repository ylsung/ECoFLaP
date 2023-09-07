
class T5BasePruner(BasePruner):
    def __init__(
        self,
        model,
        data_loader,
        task_type="vision_language",
        prune_spec=None,
        importance_scores_cache=None,
        keep_indices_or_masks_cache=None,
        is_strct_pruning=False,
        num_samples=16,
        is_global=False,
        model_prefix="t5_model",
    ):
        self.model = model
        self.data_loader = data_loader

        self.is_strct_pruning = is_strct_pruning
        self.task_type = task_type
        
        self.keep_indices_or_masks_cache = keep_indices_or_masks_cache
        self.importance_scores_cache = importance_scores_cache
        self.prune_spec = prune_spec
        self.is_global = is_global
        self.num_samples = num_samples
        self.model_prefix = model_prefix
        
        self.pruning_func = t5_strct_pruning if self.is_strct_pruning else t5_unstrct_pruning
        
        self.loss_func = loss_language
        
        self.ignore_layers = [
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        ] # not used but may be used in the future
        for k in self.model.t5_model.state_dict():
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
                
    @print_time
    def _prune(self, model, importance_scores, keep_indices_or_masks, prune_spec, ignore_layers, is_global):
        if prune_spec is None:
            # no prune spec
            return model
        
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
    def prune(self):
        dtype_record, requires_grad_record = self.model_setup_and_record_attributes(self.model)

        device = list(self.model.parameters())[0].device
        self.model.to("cpu")

        importance_scores = None
        if self.importance_scores_cache is not None:
            # cache = self.read_cache(self.importance_scores_cache)
            importance_scores = torch.load(self.importance_scores_cache)
        else:      
            importance_scores = self.compute_importance_scores(self.model, self.data_loader, self.loss_func, device)

            start_index = len(self.model_prefix) + 1
            importance_scores = {k[start_index:]: v for k, v in importance_scores.items() if k.startswith(self.model_prefix)} # filter out some info that is not for this transformer

        keep_indices_or_masks = None
        if self.keep_indices_or_masks_cache is not None:
            keep_indices_or_masks = torch.load(self.keep_indices_or_masks_cache)

        pruned_model, keep_indices_or_masks = self._prune(
            self.model, importance_scores, keep_indices_or_masks, self.prune_spec, self.ignore_layers, self.is_global
        )
        
        # let the pruned model has the original
        self.model_reset(pruned_model, dtype_record, requires_grad_record)
        
        pruned_model.to(device)
        
        return pruned_model, keep_indices_or_masks
