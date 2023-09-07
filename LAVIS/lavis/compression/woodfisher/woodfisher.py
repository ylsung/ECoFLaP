import torch
import torch.nn as nn

from copy import deepcopy
from lavis.datasets.data_utils import prepare_sample


class Model(nn.Module):
    def __init__(self, di, do):
        super().__init__()

        d1 = 8
        d2 = 6
        self.fc0 = nn.Linear(di, d1)
        self.fc1 = nn.Linear(d1, d2)
        self.fc2 = nn.Linear(d2, d1)
        self.fc3 = nn.Linear(d1, d2)
        self.fc4 = nn.Linear(d2, d1)
        self.fc5 = nn.Linear(d1, do)
    
    def forward(self, x):
        out_0 = self.fc0(x)
        out_2 = self.fc2(self.fc1(out_0)) + out_0
        out_4 = self.fc4(self.fc3(out_2)) + out_2

        return self.fc5(out_4)


def flatten_tensor_list(tensors):
    flattened = []
    for tensor in tensors:
        flattened.append(tensor.view(-1))
    return torch.cat(flattened, 0)


def _add_outer_products_efficient_v1(mat, vec, num_parts=2):
    piece = int(math.ceil(len(vec) / num_parts))
    vec_len = len(vec)
    for i in range(num_parts):
        for j in range(num_parts):
            mat[i * piece:min((i + 1) * piece, vec_len), j * piece:min((j + 1) * piece, vec_len)].add_(
                torch.ger(vec[i * piece:min((i + 1) * piece, vec_len)],
                            vec[j * piece:min((j + 1) * piece, vec_len)])
            )


class WoodFisher:
    def __init__(self, model, dloader, loss_type="cc3m", num_samples=64, size_split_group=2, fisher_damp=1e-3, fisher_parts=5, scale_prune_update=1.0, fisher_optimized=False, ignore_keys=[], to_cpu=True, dtype=torch.bfloat16):

        self.model = model
        self.dloader = dloader
        self.num_samples = num_samples

        self.size_split_group = size_split_group

        self.fisher_damp = fisher_damp
        self.fisher_parts = fisher_parts
        self.fisher_optimized = fisher_optimized
        self.scale_prune_update = scale_prune_update

        if to_cpu:
            self.device = "cpu"
        else:
            self.device = list(model.parameters())[0].device

        self.dtype = dtype

        self.groups = {}
        
        self.ignore_keys = ignore_keys

        if loss_type == "cc3m":
            self.loss_func = self.loss_cc3m
        elif loss_type == "c4":
            self.loss_func = self.loss_c4
        elif loss_type == "vit":
            self.loss_func = self.loss_vit
        else:
            raise NotImplementedError

        # record the requires_grad variable, for recovering it later
        self.requires_grad_record = {n: p.requires_grad for n, p in self.model.named_parameters()}

        self.dtype_record = {n: p.dtype for n, p in self.model.state_dict().items()}

        self.params = []

        idx = 0

        weight_idx = 0

        for name, param in model.named_parameters():
            
            if any([k in name for k in ignore_keys]):
                continue

            split_start = 0

            group_starts = []
            group_ends = []

            fisher_invs = []
            while split_start < param.numel():
                split_end = min(split_start + size_split_group, param.numel())

                group_starts.append(split_start)
                group_ends.append(split_end)
                fisher_invs.append(None)

                split_start = split_end
            
            assert split_end == param.numel()

            # self.groups[name] = [idx, weight_idx, weight_idx + param.numel(), None] # (start, end, fisher)
            self.groups[name] = [idx, group_starts, group_ends, fisher_invs] # (start, end, fisher)

            idx += 1
            weight_idx += param.numel()

            self.params.append(param)
        
        # flatten_tensor_list(params)

    def unwrap_dist_model(self, model):
        if getattr(model, "module", None) is not None:
            return model.module
        else:
            return model

    def compute_sample_fisher(self, params, loss, return_outer_product=True):
        ys = loss

        grads = torch.autograd.grad(ys, params)  # first order gradient

        assert len(grads) == len(params)

        if not return_outer_product:
            return grads
        else:
            return torch.outer(grads, grads)

    def sherman_morrison(self, idx, sample_grads, _fisher_inv):

        if idx == 0:
            numerator_normalization = (self.fisher_damp) ** 2

            # rewrite in terms of inplace operations
            _fisher_inv = torch.ger(sample_grads, sample_grads).mul_(1.0 / numerator_normalization).div_(
                self.num_samples + (sample_grads.dot(sample_grads) / self.fisher_damp)
            )
            _fisher_inv.diagonal().sub_(1.0 / self.fisher_damp)
            _fisher_inv.mul_(-1)
        
        else:
            cache_matmul = torch.matmul(_fisher_inv, sample_grads)
            cache_matmul.div_((self.num_samples + sample_grads.dot(cache_matmul)) ** 0.5)
            if not self.fisher_optimized:
                _fisher_inv.sub_(
                    torch.ger(cache_matmul, cache_matmul)
                )
            else:
                assert self.fisher_parts > 1
                # F = F - x x'
                # F1 = -F
                _fisher_inv.mul_(-1)
                # F1 + x x'
                _add_outer_products_efficient_v1(
                    _fisher_inv, cache_matmul, num_parts=self.fisher_parts
                )
                # F = - F1
                _fisher_inv.mul_(-1)

        return _fisher_inv.detach()

    def loss_cc3m(self, model, samples, cuda_enabled):
        samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

        # samples = {key: s.half() for key, s in samples.items()}

        loss_dict = model(samples)
        loss = loss_dict["loss"]

        batch_len = len(samples["text_input"])

        return loss, batch_len

    def compute_fisher_inverse(self):
        model = self.model
        device = list(model.parameters())[0].device

        accum_samples = 0
        for i, d in enumerate(self.dloader):
            print(accum_samples)
            if accum_samples >= self.num_samples:
                break

            loss, batch_len = self.loss_func(model, d, device != "cpu")

            accum_samples += batch_len

            sample_grads = self.compute_sample_fisher(self.params, loss, return_outer_product=False)

            for k in self.groups:
                print(k)
                idx, starts, ends, fisher_invs = self.groups[k]

                flatten_param = flatten_tensor_list(sample_grads[idx]).type(self.dtype)

                for list_id, (start, end, fisher_inv) in enumerate(zip(starts, ends, fisher_invs)):
                    split_flatten_param = flatten_param[start: end]

                    if fisher_inv is not None:
                        fisher_inv = fisher_inv.to(flatten_param.device)

                    fisher_inv = self.sherman_morrison(i, split_flatten_param, fisher_inv)

                    self.groups[k][-1][list_id] = fisher_inv.to(self.device)

        # for k, v in self.groups.items():
        #     print(k)

        #     for _v in v[-1]:
        #         print(_v.shape)

        return self.groups

    def compute_importance_scores(self, param, block_fisher_inv_diag, subtract_min=False):
        if param is None: return None
        # w_i **2 x ((F)^-1)_ii,
        inv_fisher_diag_entry = block_fisher_inv_diag.view_as(param).to(param.device)
        # print(f"mean value of statistic without eps = {1e-10} is {(torch.mean((param ** 2)/inv_fisher_diag_entry)).item()}")
        # print(f"std value of statistic without eps = {1e-10} is {(torch.std((param ** 2) / inv_fisher_diag_entry)).item()}")

        # multiplying by the current mask makes the corresponding statistic
        # of those weights zero and keeps them removed.
        # print(f'mean value of param^2 is {(param**2).mean().item()} and std is {(param**2).std().item()}')
        # print(f'mean value of inv fisher is {inv_fisher_diag_entry.mean().item()} and std is {inv_fisher_diag_entry.std().item()}')
        optimal_brain_damage_stat = (param ** 2)/(inv_fisher_diag_entry + 1e-10)

        if subtract_min:
            # print('subtracting min in param_stat')
            optimal_brain_damage_stat = optimal_brain_damage_stat - optimal_brain_damage_stat.min()

        pruning_stat = optimal_brain_damage_stat + 1e-10

        return pruning_stat

    def unstrct_pruning(self, importance_measure, ratio):
        masks = {}

        for k, v in importance_measure.items():
            top_k = int(v.numel() * ratio)

            _, top_indices = v.float().reshape(-1).topk(top_k, dim=-1)

            mask = torch.zeros((v.numel(),), dtype=bool, device=v.device) # 1D
            mask.scatter_(-1, top_indices, 1)

            mask = mask.reshape_as(v)

            masks[k] = mask

        return masks

    def model_setup(self):
        for n, p in self.model.state_dict().items():
            p.data = p.data.type(torch.bfloat16)

        # set requires_grad to be true for getting model's derivatives
        for n, p in self.model.named_parameters():
            p.requires_grad = True

    def model_reset(self):
        # set to original requires grad
        for n, p in self.model.named_parameters():
            p.requires_grad = self.requires_grad_record[n]

        for n, p in self.model.state_dict().items():
            p.data = p.data.type(self.dtype_record[n])

    def compute_fisher_inv_and_importance_score(self):

        self.model_setup()

        print("compute fisher inv")
        # compute the inverse fisher
        self.compute_fisher_inverse()

        # retrieve the inverse diagonal fisher

        print("compute diag fisher")

        fisher_inv_diag_group = {}

        for name, group in self.groups.items():
            fisher_invs_diag = []
            fisher_invs = group[-1]
            for fisher_inv in fisher_invs:
                fisher_invs_diag.append(fisher_inv.diagonal())

            fisher_invs_diag = torch.cat(fisher_invs_diag)

            fisher_inv_diag_group[name] = fisher_invs_diag

        importance_scores = {}

        print("compute importance score")

        for name, fisher_inv_diag in fisher_inv_diag_group.items():
            idx = self.groups[name][0]
            param = self.params[idx].type(self.dtype).to(self.device)
            importance_scores[name] = self.compute_importance_scores(param, fisher_inv_diag)

            # print(name, importance_scores[name].shape)

        self.fisher_inv_diag_group = fisher_inv_diag_group

        self.model_reset()

        return importance_scores

    def reweighting_after_pruning(self, original_weights, keep_masks):
        self.model_setup()

        new_weights = {}
        for key, (idx, split_starts, split_ends, fisher_invs) in self.groups.items():
            keep_mask = keep_masks[key].type(self.dtype).to(self.device)
            pruned_mask = 1 - keep_mask
            param = self.params[idx].type(self.dtype).to(self.device)
            fisher_inv_diag = self.fisher_inv_diag_group[key]
            scale_basis = self.get_pruned_wts_scaled_basis(pruned_mask.flatten(), param.flatten(), fisher_inv_diag)
            weight_update = self.get_weights_update(split_starts, split_ends, fisher_invs, scale_basis)

            # print(weight_update.sum())

            # print(scale_basis.sum())
            
            w_device = original_weights[key].device
            w_type = original_weights[key].dtype
            new_weights[key] = (
                original_weights[key] + weight_update.view_as(original_weights[key]).type(w_type).to(w_device)
                ) * keep_mask.type(w_type).to(w_device)

            # print(pruned_mask.sum() / pruned_mask.numel())

            # print(new_weights[key].sum())

        for key in original_weights:
            if any([k in key for k in self.ignore_keys]):
                # not pruned 
                new_weights[key] = original_weights[key]

        self.model_reset()

        return new_weights

    def get_weights_update(self, split_starts, split_ends, fisher_invs, scaled_basis_vector):
        weight_update = []
        split_start = 0

        for split_start, split_end, fisher_inv in zip(split_starts, split_ends, fisher_invs):
            weight_update.append(
                fisher_inv @ scaled_basis_vector[split_start: split_end]
            )

        assert split_end == scaled_basis_vector.shape[0]
        weight_update = torch.cat(weight_update)

        # "scale_prune_update": reduces the effect of weight update.
        # Also, it is okay to not worry about the parameters that are going to be removed
        # as they will be anyways masked.
        weight_update = self.scale_prune_update * weight_update

        return weight_update

    def get_pruned_wts_scaled_basis(self, pruned_params, flattened_params, fisher_inv_diag):
        return -1 * torch.div(torch.mul(pruned_params, flattened_params), fisher_inv_diag)
