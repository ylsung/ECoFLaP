import torch
import torch.nn as nn
import numpy as np


batch_size = 60000
dim_1 = 700
dim_2 = 700
x1 = torch.randn(batch_size, dim_1)

l1 = nn.Linear(dim_1, dim_2, bias=False)

l2 = nn.Linear(dim_1, dim_2, bias=False)

l3 = nn.Linear(dim_1, dim_2, bias=False)


x2 = l1(x1)
x3 = l2(x2)


# x2 = torch.randn(batch_size, dim_1)
# y1 = l1(x1)
# y2 = l2(x2)

print("Interpolation")

# def compute_diff(merge_l, l1, l2, x1, x2):
#     return .5 * torch.norm(l2(x2) - merge_l(x2), dim=-1).mean() + .5 * torch.norm(l1(x1) - merge_l(x1), dim=-1).mean()

# def compute_diff(merge_l, l1, l2, x1, x2):
#     return .5 * torch.norm(l2(x2) - merge_l(x2), dim=-1).mean() + .5 * torch.norm(l1(x1) - merge_l(x1), dim=-1).mean()

def compute_diff(merge_l, l1, l2, x1, x2):
    return torch.norm(l2(l1(x1)) - merge_l(x1), dim=-1).mean()

def compute_diff_2(merge_l, l1, l2, x1, x2):
    return torch.norm(l2(l1(x1)) - merge_l(merge_l(x1)), dim=-1).mean()


for ratio in np.linspace(0, 1, 11):

    l3.weight.data = ratio * l1.weight.data + (1 - ratio) * l2.weight.data

    # gen_x3 = l3(x1)

    # print(ratio, torch.norm(x3 - gen_x3, dim=-1).mean())
    diff = compute_diff_2(l3, l1, l2, x1, x2)
    
    print(ratio, diff)

print("RegMean")

gram_1 = torch.matmul(x1.T, x1)
gram_2 = torch.matmul(x2.T, x2)

def rescale_gram_matrice(gram, scaling_factor):
    scaling_for_non_diag = scaling_factor
    diag = torch.diag_embed(torch.diag(gram))

    return scaling_for_non_diag * gram + (1 - scaling_for_non_diag) * diag

for ratio in np.linspace(0, 1, 11):
    r_gram_1 = rescale_gram_matrice(gram_1, ratio)
    r_gram_2 = rescale_gram_matrice(gram_2, ratio)
    weight_to_merge = torch.matmul(l1.weight.data, r_gram_1) + torch.matmul(l2.weight.data, r_gram_2)

    total_gram = r_gram_1 + r_gram_2

    normalization_factor = torch.inverse(total_gram)
    l3.weight.data = torch.matmul(weight_to_merge, normalization_factor)

    diff_1 = compute_diff_2(l3, l1, l2, x1, x2)

    l3.weight.data = torch.linalg.lstsq(total_gram.T, weight_to_merge.T).solution.T

    diff_2 = compute_diff_2(l3, l1, l2, x1, x2)
    
    print(ratio, diff_1, diff_2)


print("RegMean for Distillation")

gram_1 = torch.matmul(x1.T, x1)
gram_2 = torch.matmul(x2.T, x2)

gram_1_3 = torch.matmul(x3.T, x1)

def rescale_gram_matrice(gram, scaling_factor):
    scaling_for_non_diag = scaling_factor
    diag = torch.diag_embed(torch.diag(gram))

    return scaling_for_non_diag * gram + (1 - scaling_for_non_diag) * diag

for ratio in np.linspace(0, 1, 11):
    reg_1 = rescale_gram_matrice(gram_1, 0)
    reg_2 = rescale_gram_matrice(gram_2, 0)
    r_gram_1 = gram_1 + reg_1
    r_gram_2 = gram_2 + reg_2
    weight_to_merge = torch.matmul(l1.weight.data, reg_1) + torch.matmul(l2.weight.data, reg_2) + ratio * gram_1_3

    total_gram = r_gram_1 + r_gram_2

    normalization_factor = torch.inverse(total_gram)
    l3.weight.data = torch.matmul(weight_to_merge, normalization_factor)

    diff_1 = compute_diff(l3, l1, l2, x1, x2)

    l3.weight.data = torch.linalg.lstsq(total_gram.T, weight_to_merge.T).solution.T

    diff_2 = compute_diff(l3, l1, l2, x1, x2)
    
    print(ratio, diff_1, diff_2)