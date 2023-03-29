import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import functools


x = torch.randn(50, 30)
y = torch.randn(50, 30)

l_list = nn.Sequential(
    *[nn.Linear(30, 30) for i in range(5)]
)

merge_weights = nn.Parameter(torch.randn(5))

merge_l = nn.Linear(30, 30)

merge_l.weight_list = torch.stack(
    [l.weight.detach() for l in l_list]
)

print(merge_l.weight_list)

print(merge_l.weight_list.shape)

merge_l.bias_list = torch.stack(
    [l.bias.detach() for l in l_list]
)

opt = torch.optim.Adam([merge_weights], lr=0.08)

def linear_forward(self, input):
    # print(merge_weights.shape, self.weight_list.shape)
    weight = torch.einsum("b,bjk->jk", merge_weights, self.weight_list)
    # print(torch.all(weight == self.weight_list[0] + 2*self.weight_list[2]))
    bias = torch.einsum("b,bj->j", merge_weights, self.bias_list)
    # print(torch.all(bias == self.bias_list[0] + 2*self.bias_list[2]))
    return F.linear(input, weight, bias)

merge_l.forward = functools.partial(linear_forward, merge_l)

print(merge_l(x).shape)


for i in range(100):
    p = merge_l(x)

    loss = ((p - y) ** 2).mean()

    print(loss)
    # print(merge_weights)

    opt.zero_grad()

    loss.backward()

    opt.step()
