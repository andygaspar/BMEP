import copy
import random
import time

import torch

from Data_.data_loader import DistanceData
from Solvers.Random.random_solver import RandomSolver


def generate_adj(dim, solver, batch_size=1):
    adj_mats = torch.tensor(solver.initial_adj_mat(n_problems=batch_size)).to('cpu')
    for step in range(3, dim):
        choices = 3 + (step - 3) * 2
        idxs_list = torch.nonzero(torch.triu(adj_mats)).reshape(batch_size, -1, 3)
        rand_idxs = random.choices(range(choices), k=batch_size)
        idxs_list = idxs_list[[range(batch_size)], rand_idxs, :]
        idxs_list = (idxs_list[:, :, 0], idxs_list[:, :, 1], idxs_list[:, :, 2])
        adj_mats = solver.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=dim)

    return adj_mats


def split_tree(i):
    ii = i

    a = torch.zeros(dim - 2)
    b = torch.ones(dim - 2)
    b[i] = 0

    g = torch.nonzero(ad_int[i], as_tuple=True)[-1]
    a[g[0]] = 1
    idx = g[0]
    ad_int[ii, :] = ad_int[:, ii] = 0
    ii = idx
    for _ in range(dim - 3):
        idx = torch.nonzero(ad_int[idx]).T[-1]
        a[idx] = 1
        ad_int[ii, :] = ad_int[:, ii] = 0
        ii = idx

    b -= a
    return a, b

distances = DistanceData()
distances.print_dataset_names()
data_set = distances.get_dataset(3)
dim = 10

d = data_set.get_random_mat(dim)
solver = RandomSolver(d)

batch_size = 1


adj_mats = generate_adj(dim, solver, batch_size)
r = time.time()
print(time.time() - r)

adj_mat = adj_mats.squeeze(0)

ad_int = copy.deepcopy(adj_mat[dim:, dim:])

i = random.choices(range(dim-2))[0]


print("split", i)
print(adj_mat[dim:, dim:])

a, b = split_tree(i)

print(a, '\n', b)

