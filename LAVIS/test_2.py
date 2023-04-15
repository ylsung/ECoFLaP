import torch
import random

def get_T(index):
    T = torch.zeros(len(index), len(index))
    T[torch.arange(len(index)), index] = 1
    return T

def permute_T(w, T, axis):
    # axis = len(w.shape) - 1 if axis == -1 else axis

    T = T.t()

    w = torch.matmul(torch.transpose(w, axis, -1), T).transpose(-1, axis)

    return w


for dim in [2, 3, 4]:

    dims = [random.randint(40, 60) for i in range(dim)]

    a = torch.randn(*dims)

    for i in range(dim):

        # if i == dim - 1:
        #     i == -1

        index = torch.randperm(a.shape[i])

        gt = torch.index_select(a, i, index.int())

        T = get_T(index)

        pr = permute_T(a, T, i)

        try:

            assert torch.all(gt == pr)
        
        except:
            # print(dim, i)
            print(gt)
            print(pr)

            print(a)
            print(index)

# axis = len(w.shape) - 1 if axis == -1 else axis

# T = perm[p] if axis % 2 == 0 else perm[p].t()

# w = torch.matmul(torch.transpose(w, axis, -1), T).transpose(-1, axis)