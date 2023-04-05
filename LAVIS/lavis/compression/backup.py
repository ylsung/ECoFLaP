import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.t5.modeling_t5 import T5LayerNorm


def get_init_weight_for_merge(size_of_layer):
    return nn.Parameter(torch.full((size_of_layer, ), 1 / size_of_layer))


class MergeWeightModule(nn.Module):
    def __init__(self, size_of_layer, weight_type):
        if weight_type == "scalar":
            self.weight = nn.Parameter(torch.full((size_of_layer, ), 1 / size_of_layer))

    def get_weight(self):
        if weight_type == "scalar"
            return self.weight


class LearnableMergeModule(nn.Module):
    def __init__(self, layers, merge_weights=None, weight_type="scalar"):
        super().__init__()
        self.merge_weights = MergeWeightModule(len(layers))
        self.candidate_weight = nn.Parameter(torch.stack(
            [l.weight.detach() for l in layers]
        ))

        if getattr(layers[0], "bias", None) is not None:
            self.candidate_bias = nn.Parameter(torch.stack(
                [l.bias.detach() for l in layers]
            ))
        else:
            self.candidate_bias = None

        self.num_candidates = len(layers)

        # self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.merge_weights, a=math.sqrt(5))


class LearnableMergeLinear(LearnableMergeModule):
    def __init__(self, layers, merge_weights=None, weight_type="scalar"):
        super().__init__(layers, merge_weights)
        self.get_orig_attributes(layers[0])

    def get_orig_attributes(self, layer):
        self.in_features = layer.in_features
        self.out_features = layer.out_features

    def forward(self, input):
        weight = torch.einsum("b,bjk->jk", self.merge_weights, self.candidate_weight)
        # print(torch.all(weight == self.weight_list[0] + 2*self.weight_list[2]))

        if self.candidate_bias is not None:
            bias = torch.einsum("b,bj->j", self.merge_weights, self.candidate_bias)
        else:
            bias = None
        # print(torch.all(bias == self.bias_list[0] + 2*self.bias_list[2]))
        # print("linear forward")
        return F.linear(input, weight, bias)

    def extra_repr(self) -> str:
        return 'num_candidates={}, in_features={}, out_features={}, bias={}'.format(
            self.num_candidates, self.in_features, self.out_features, self.candidate_bias is not None
        )

    def convert_to_normal_save_weights(self):
        self.weight = nn.Parameter(torch.einsum("b,bjk->jk", self.merge_weights, self.candidate_weight))
        # print(torch.all(weight == self.weight_list[0] + 2*self.weight_list[2]))

        if self.candidate_bias is not None:
            self.bias = nn.Parameter(torch.einsum("b,bj->j", self.merge_weights, self.candidate_bias))
        else:
            self.bias = None


class LearnableMergeLayerNorm(LearnableMergeModule):
    def __init__(self, layers, merge_weights=None, weight_type="scalar"):
        super().__init__(layers, merge_weights)
        self.get_orig_attributes(layers[0])

    def get_orig_attributes(self, layer):
        self.normalized_shape = layer.normalized_shape  # type: ignore[arg-type]
        self.eps = layer.eps
        self.elementwise_affine = layer.elementwise_affine

    def forward(self, input):
        weight = torch.einsum("b,bj->j", self.merge_weights, self.candidate_weight)
        # print(torch.all(weight == self.weight_list[0] + 2*self.weight_list[2]))

        if self.candidate_bias is not None:
            bias = torch.einsum("b,bj->j", self.merge_weights, self.candidate_bias)
        else:
            bias = None
        # print(torch.all(bias == self.bias_list[0] + 2*self.bias_list[2]))
        # print("layernorm forward")

        return F.layer_norm(
            input, self.normalized_shape, weight, bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

    def convert_to_normal_save_weights(self):
        self.weight = nn.Parameter(torch.einsum("b,bj->j", self.merge_weights, self.candidate_weight))
        if self.candidate_bias is not None:
            self.bias = nn.Parameter(torch.einsum("b,bj->j", self.merge_weights, self.candidate_bias))
        else:
            self.bias = None


class LearnableMergeT5LayerNorm(LearnableMergeModule):
    def __init__(self, layers, merge_weights=None, weight_type="scalar"):
        super().__init__(layers, merge_weights)
        self.get_orig_attributes(layers[0])

    def get_orig_attributes(self, layer):
        self.variance_epsilon = layer.variance_epsilon

    def forward(self, hidden_states):

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        weight = torch.einsum("b,bj->j", self.merge_weights, self.candidate_weight)
        # print(torch.all(weight == self.weight_list[0] + 2*self.weight_list[2]))

        if self.candidate_bias is not None:
            bias = torch.einsum("b,bj->j", self.merge_weights, self.candidate_bias)
        else:
            bias = None
        # print(torch.all(bias == self.bias_list[0] + 2*self.bias_list[2]))
        # print("layernorm forward")

        # convert into half-precision if necessary
        if weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(weight.dtype)

        return weight * hidden_states

    def convert_to_normal_save_weights(self):
        self.weight = nn.Parameter(torch.einsum("b,bj->j", self.merge_weights, self.candidate_weight))
        if self.candidate_bias is not None:
            self.bias = nn.Parameter(torch.einsum("b,bj->j", self.merge_weights, self.candidate_bias))
        else:
            self.bias = None


def convert_to_normal_save_weights(transformer):
    for k, module in dict(transformer.named_modules()).items():
        for c_name, layer in dict(module.named_children()).items():
            if hasattr(layer, "convert_to_normal_save_weights"):
                layer.convert_to_normal_save_weights()

    for k, module in dict(transformer.named_modules()).items():
        for c_name, layer in dict(module.named_children()).items():
            if hasattr(layer, "convert_to_normal_save_weights"):
                del layer.merge_weights
                del layer.candidate_weight
                if hasattr(layer, "candidate_bias"):
                    del layer.candidate_bias


def get_proper_name_for_layer_access(name):
    # Use regular expression to match the pattern "block.*\.layer.*\."
    match = re.search(r"block\.(\d+)\.layer\.(\d+)\.", name)

    if match:
        # Replace the matched pattern with "[block_index].[layer_index]."
        block_index = match.group(1)
        layer_index = match.group(2)
        new_name = re.sub(r"block\." + block_index + "\.layer\." + layer_index + "\.", 
                            f"block[{block_index}].layer[{layer_index}].", name)
        return new_name
    else:
        raise ValueError("the name is not in a correct format")


def t5_modify_for_learnable_merge(orig_transformer, distilled_transformer, distilled_block_ids, learnable_weight_type):
    # initialize other modules first and initialize them as the first N layers
    orig_transformer_state_dict = orig_transformer.state_dict()
    weights = {}

    for k in distilled_transformer.state_dict().keys():
        weights[k] = orig_transformer.state_dict()[k]

    distilled_transformer.load_state_dict(weights)

    last_block_id = -1

    # add learnable modules
    for k, module in dict(distilled_transformer.named_modules()).items():

        current_block = re.findall("block[.][0-9]*", k)
        assert len(current_block) <= 1
        current_block = current_block[0] if len(current_block) > 0 else None

        if current_block is None or "relative_attention_bias" in k:
            continue

        current_block_id = int(current_block.split(".")[-1])

        block_ids_to_distill = distilled_block_ids[current_block_id]

        if isinstance(block_ids_to_distill, int):
            block_ids_to_distill = [block_ids_to_distill]

        block_ids_to_distill = list(block_ids_to_distill)

        print(block_ids_to_distill)

        # if len(block_ids_to_distill) == 1:
        #     # dont need to merge
        #     continue

        weight_type, share_type = learnable_weight_type.split("-")

        assert share_type in ["shared", "unshared"], "The share_type is not support"

        if share_type == "shared":
            if last_block_id != current_block_id:
                last_block_id = current_block_id
                merge_weights = get_init_weight_for_merge(len(block_ids_to_distill))
            else:
                # reuse the merge weights for the whole block
                pass
        else:
            merge_weights = None

        for c_name, layer in dict(module.named_children()).items():

            if not isinstance(layer, (nn.Linear, T5LayerNorm)):
                continue

            layers = []

            for block_id in block_ids_to_distill:
                merge_block = f"block.{block_id}"

                merge_k = k.replace(current_block, merge_block)

                merge_full_c_name = "orig_transformer." + merge_k + "." + c_name

                merge_full_c_name = get_proper_name_for_layer_access(merge_full_c_name)

                # print(k, merge_full_c_name)

                layers.append(eval(merge_full_c_name))

            if isinstance(layer, nn.Linear):
                # print("set")
                setattr(
                    module,
                    c_name,
                    LearnableMergeLinear(layers, merge_weights),
                )

            elif isinstance(layer, T5LayerNorm):
                # print("set")
                setattr(
                    module,
                    c_name,
                    LearnableMergeT5LayerNorm(layers, merge_weights),
                )

    return distilled_transformer
