import random
import time
import numpy as np
import torch

from Solvers.Random.test_precomputed_distance_torch import PrecomputeTorch
from Solvers.solver import Solver


class PrecomputeTorch3(Solver):

    def __init__(self, d, sorted_d=False, device='cuda:0'):
        super(PrecomputeTorch3, self).__init__(d, sorted_d)
        self.device = device
        self.d = torch.tensor(self.d, device=self.device)
        # self.powers = torch.tensor(self.powers, device=self.device)
        self.powers = self.powers.to(self.device)
        self.subtrees_mat = None
        self.pointer_subtrees = None
        self.subtrees_idx_mat = None
        self.subtree_dist = None
        self.T_new = None

        self.mask = torch.zeros(2*(2*self.n_taxa - 3), dtype=torch.bool, device=self.device)

    def solve(self, start=3, adj_mat=None):
        t = time.time()
        adj_mat = self.initial_adj_mat(self.device) if adj_mat is None else adj_mat
        subtrees_mat, subtrees_idx_mat  = self.initial_sub_tree_mat()
        T = self.init_tau()
        s_mat_step = 6
        # subtree_dist = self.init_sub_dist()
        for step in range(start, self.n_taxa):
            choices = 3 + (step - 3) * 2
            idxs_list = torch.nonzero(torch.triu(adj_mat))
            rand_idxs = random.choices(range(choices))
            idxs_list = idxs_list[rand_idxs][0]
            adj_mat = self.add_node(adj_mat, idxs_list, step, self.n_taxa)

            subtrees_mat, subtrees_idx_mat, T= \
                self.add_subtrees(subtrees_mat, subtrees_idx_mat, step, idxs_list, s_mat_step, T)
            s_mat_step += 4

        s = subtrees_mat[:, :self.n_taxa].to(torch.float64)
        # z = torch.matmul(s, self.d * self.powers[T[:n, :n]])
        self.subtree_dist = torch.matmul(torch.matmul(s, self.d * self.powers[T[:n, :n]]), s.T) * 2
        self.subtrees_mat = subtrees_mat
        self.subtrees_idx_mat = subtrees_idx_mat
        self.solution = adj_mat
        # self.obj_val = self.compute_obj_val_from_adj_mat(adj_mat, self.d, self.n_taxa)
        # print(T[:self.n_taxa, :self.n_taxa])
        # self.T = self.get_full_tau(self.solution.to('cpu'))
        self.T_new = T
        # self.device = 'cuda:0'
        print('build time', time.time() - t)
        # t = time.time()

        # print(time.time() - t)


    def initial_sub_tree_mat(self):
        subtree_mat = torch.zeros((2*(2*self.n_taxa - 3), self.m), device=self.device, dtype=torch.long)
        idx_subtrees_mat = torch.zeros((self.m, self.m), device=self.device, dtype=torch.long)

        # 0
        subtree_mat[0, 0]  = 1
        subtree_mat[1, 1]  = 1
        subtree_mat[2, 2]  = 1

        idx_subtrees_mat[self.n_taxa, 0] = 0
        idx_subtrees_mat[self.n_taxa, 1] = 1
        idx_subtrees_mat[self.n_taxa, 2] = 2

        # 1
        subtree_mat[3, self.n_taxa] = subtree_mat[3, 1] = subtree_mat[3, 2] = 1
        subtree_mat[4, self.n_taxa] = subtree_mat[4, 0] = subtree_mat[4, 2] = 1
        subtree_mat[5, self.n_taxa] = subtree_mat[5, 0] = subtree_mat[5, 1] = 1


        idx_subtrees_mat[0, self.n_taxa] = 3
        idx_subtrees_mat[1, self.n_taxa] = 4
        idx_subtrees_mat[2, self.n_taxa] = 5

        return  subtree_mat, idx_subtrees_mat
    def add_subtrees(self, subtree_mat, idx_subtrees_mat, new_taxon_idx, idx, s_mat_step, T):

        new_internal_idx = self.n_taxa + new_taxon_idx - 2

        # singleton {k}
        subtree_mat[s_mat_step, new_taxon_idx] = 1
        idx_subtrees_mat[new_internal_idx, new_taxon_idx] = s_mat_step

        # {k} complementary
        subtree_mat[s_mat_step + 1, :new_taxon_idx] = 1
        subtree_mat[s_mat_step + 1, self.n_taxa: new_internal_idx + 1] = 1
        idx_subtrees_mat[new_taxon_idx, new_internal_idx] = s_mat_step + 11

        # update T i->j j->i distance *********************

        ij = subtree_mat[idx_subtrees_mat[idx[0], idx[1]]]
        ji = subtree_mat[idx_subtrees_mat[idx[1], idx[0]]]

        T[ij == 1] += ji
        T[ji == 1] += ij

        T[new_taxon_idx] = T[idx[0]] * ij + T[idx[1]] * ji


        # add distance k and internal k
        T[:, new_taxon_idx] = T[new_taxon_idx]
        T[new_internal_idx, :new_taxon_idx + 1] = T[:new_taxon_idx + 1, new_internal_idx] \
            = T[new_taxon_idx, :new_taxon_idx + 1] - 1
        T[new_internal_idx, self.n_taxa:new_internal_idx + 1] = T[self.n_taxa:new_internal_idx + 1, new_internal_idx] \
            = T[new_taxon_idx, self.n_taxa:new_internal_idx + 1] - 1
        T[new_internal_idx, new_internal_idx] = 0
        T[new_taxon_idx, new_internal_idx] = T[new_internal_idx, new_taxon_idx] = 1



        # i -> j to k -> j
        subtree_mat[s_mat_step + 2] = ij
        idx_subtrees_mat[new_internal_idx, idx[1]] = s_mat_step + 2


        # j -> i to k -> i
        subtree_mat[s_mat_step + 3] = ji
        idx_subtrees_mat[new_internal_idx, idx[0]] = s_mat_step + 3

        # add k and new internal to previous

        subtree_mat[:s_mat_step, new_taxon_idx] = subtree_mat[:s_mat_step, new_internal_idx] = \
            subtree_mat[:s_mat_step, idx[0]] + subtree_mat[:s_mat_step, idx[1]] - \
            (subtree_mat[:s_mat_step, idx[0]] * subtree_mat[:s_mat_step, idx[1]])


        # adjust idxs
        idx_subtrees_mat[idx[0], new_internal_idx] = idx_subtrees_mat[idx[0], idx[1]]
        idx_subtrees_mat[idx[1], new_internal_idx] = idx_subtrees_mat[idx[1], idx[0]]
        idx_subtrees_mat[idx[0], idx[1]] = idx_subtrees_mat[idx[1], idx[0]] = 0

        return subtree_mat, idx_subtrees_mat, T


    def init_sub_dist(self):
        subtree_dist = torch.zeros((self.m, self.m), device=self.device, dtype=torch.bool)
        subtree_dist[:self.n_taxa, :self.n_taxa] = self.d

        return subtree_dist

    def init_tau(self):
        T = torch.zeros((self.m, self.m), dtype=torch.long, device=self.device)
        T[0, 1] = T[0, 2] = T[1, 0] = T[1, 2] = T[2, 0] = T[2, 1] = 2
        T[0, self.n_taxa] = T[1, self.n_taxa] = T[2, self.n_taxa] =  \
            T[self.n_taxa, 0] = T[self.n_taxa, 1] = T[self.n_taxa, 2] = 1

        return T

    def check_dist(self):
        val = self.d * self.powers[self.T_new[:self.n_taxa, :self.n_taxa]]
        dist = torch.zeros_like(self.subtree_dist)
        for i in range(self.subtree_dist.shape[0]):
            for j in range(self.subtree_dist.shape[0]):
                a = self.subtrees_mat[i, :self.n_taxa].float()
                b = self.subtrees_mat[j, :self.n_taxa].float()
                idx = torch.matmul(a.unsqueeze(0).T, b.unsqueeze(0)).to(torch.long)==1
                v = val[idx].flatten().sum() * (1 - (self.subtrees_mat[i] * self.subtrees_mat[j]).sum()>0)*2
                dist[i, j] = v
        return dist
#ll

torch.set_printoptions(precision=2, linewidth=150)

random.seed(0)
np.random.seed(0)

n = 40

d = np.random.uniform(0,1,(n, n))
d = np.triu(d) + np.triu(d).T
np.fill_diagonal(d, 0)

device = 'cpu'
# device = 'cuda:0'

model = PrecomputeTorch3(d, device=device)
t = time.time()

model.solve()
print(time.time() - t)

# s = model.subtree_dist * (torch.matmul(model.subtrees_mat.to(torch.float), model.subtrees_mat.to(torch.float).T) == 0)
# print(torch.allclose(model.check_dist(), s))
print()
# sub = model.subtrees_mat
# sub.repeat((sub.shape[0]), 1)
# sub_int = sub.repeat_interleave(sub.shape[0], dim=0)
# dd = model.d
# z=torch.matmul(sub[:, :4].to(torch.float64), dd*model.powers[model.T_new[:n, :n]])
# h = torch.matmul(z, sub[:, :4].to(torch.float64).T)*20
# print(torch.nonzero(model.solution))
# for el in torch.nonzero(model.solution):
#     print(el, torch.nonzero(model.subtrees_mat[model.subtrees_idx_mat[el[0], el[1]]]).T)
# print(torch.equal(torch.tensor(model.T).to('cuda:0'), model.T_new))
# print(model.T)
# print(model.T_new)


# t = time.time()
# model = PrecomputeTorch(d)
# model.solve()
# print(time.time() - t)
# print(torch.nonzero(model.solution))
# print(model.T)





# print(model.subtrees_mat)


# for i in range(model.m):
#     for j in range(model.m):
#         if torch.nonzero(model.subtrees_mat[i, j]).T.squeeze(0).shape[0] > 0:
#             print(i, j, torch.nonzero(model.subtrees_mat[i, j]).T.squeeze(0))
# print(torch.nonzero(model.solution).T.squeeze(0))
#
# print(model.subtree_dist)

# for i in range(model.m):
#     for j in range(model.m):
#         if torch.nonzero(model.subtrees_mat[i, j]).T.squeeze(0).shape[0] > 0:
#             print(i, j, torch.nonzero(model.subtrees_mat[i, j]).T.squeeze(0))


# import torch
# d = torch.rand((3, 3)).triu()
# d = d + d.T
# x = torch.tensor([[1, 1, 0],[0,0,1]])
# a =x.repeat(2,1)
# b = x.repeat_interleave(2, dim=0)
# res = (1 - (a*b).sum(dim=-1))
# res[res<0] = 0
# s = torch.matmul(x.flatten().unsqueeze(1), x.flatten().unsqueeze(0))
#
# dist = torch.stack((d.repeat(2, 2) * s).split(3, dim=-1)).view(-1, 3, 3).transpose(1,2)
# f = (dist.reshape(-1, 3*3).sum(dim=-1) * res).reshape((2, 2))




