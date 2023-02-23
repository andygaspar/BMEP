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
        self.powers = torch.tensor(self.powers, device=self.device)
        self.subtrees_mat = None
        self.subtrees_list = None
        self.subtrees_idx_mat = None
        self.subtree_dist = None
        self.T_new = None

        self.mask = torch.zeros(2*(2*self.n_taxa - 3), dtype=torch.bool, device=self.device)

    def solve(self, start=3, adj_mat=None):
        t = time.time()
        adj_mat = self.initial_adj_mat(self.device) if adj_mat is None else adj_mat
        subtrees_mat, subtrees_idx_mat, subtrees_dist_mat, subtrees_list = self.initial_sub_tree_mat()
        T = self.init_tau()
        s_mat_step = 6
        # subtree_dist = self.init_sub_dist()
        for step in range(start, self.n_taxa):
            choices = 3 + (step - 3) * 2
            idxs_list = torch.nonzero(torch.triu(adj_mat))
            rand_idxs = random.choices(range(choices))
            idxs_list = idxs_list[rand_idxs][0]
            adj_mat = self.add_node(adj_mat, idxs_list, step, self.n_taxa)

            subtrees_mat, subtrees_idx_mat, T, subtrees_dist_mat= \
                self.add_subtrees(subtrees_mat, subtrees_idx_mat, subtrees_dist_mat, subtrees_list, step, idxs_list, s_mat_step, T)
            s_mat_step += 4

        self.subtrees_mat = subtrees_mat
        self.subtrees_idx_mat = subtrees_idx_mat
        self.subtree_dist = subtrees_dist_mat
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
        subtree_idx_mat = torch.zeros((self.m, self.m), device=self.device, dtype=torch.long)
        subtrees_dist_mat = torch.zeros((2*(2*self.n_taxa - 3), 2*(2*self.n_taxa - 3)), device=self.device, dtype=torch.float64)
        subtrees_list = torch.zeros((2 * (2 * self.n_taxa - 3), 2 * (2 * self.n_taxa - 3)), device=self.device,
                                        dtype=torch.float64)

        # 0
        subtree_mat[0, 0]  = 1
        subtree_mat[1, 1]  = 1
        subtree_mat[2, 2]  = 1

        subtree_idx_mat[self.n_taxa, 0] = 0
        subtree_idx_mat[self.n_taxa, 1] = 1
        subtree_idx_mat[self.n_taxa, 2] = 2

        subtrees_dist_mat[0, 1] = subtrees_dist_mat[1, 0] = self.d[0, 1]/2
        subtrees_dist_mat[0, 2] = subtrees_dist_mat[2, 0] = self.d[0, 2]/2
        subtrees_dist_mat[1, 2] = subtrees_dist_mat[2, 1] = self.d[1, 2]/2

        # 1
        subtree_mat[3, self.n_taxa] = subtree_mat[3, 1] = subtree_mat[3, 2] = 1
        subtree_mat[4, self.n_taxa] = subtree_mat[4, 0] = subtree_mat[4, 2] = 1
        subtree_mat[5, self.n_taxa] = subtree_mat[5, 0] = subtree_mat[5, 1] = 1


        subtree_idx_mat[0, self.n_taxa] = 3
        subtree_idx_mat[1, self.n_taxa] = 4
        subtree_idx_mat[2, self.n_taxa] = 5

        subtrees_dist_mat[0, 3] = subtrees_dist_mat[3, 0] = (self.d[0, 1] + self.d[0, 2])/2
        subtrees_dist_mat[1, 4] = subtrees_dist_mat[4, 1] = (self.d[1, 0] + self.d[1, 2])/2
        subtrees_dist_mat[2, 5] = subtrees_dist_mat[5, 2] = (self.d[2, 0] + self.d[2, 1])/2


        subtrees_list[0, 0] = 1
        subtrees_list[1, 1] = 1
        subtrees_list[2, 2] = 1
        subtrees_list[3, 1] = subtrees_list[3, 2] = subtrees_list[3, 3] = 1
        subtrees_list[4, 0] = subtrees_list[4, 2] = subtrees_list[4, 4] = 1
        subtrees_list[5, 0] = subtrees_list[5, 1] = subtrees_list[5, 5] = 1



        return  subtree_mat, subtree_idx_mat, subtrees_dist_mat, subtrees_list

    def add_subtrees(self, subtree_mat, subtree_idx_mat, subtrees_dist_mat, subtrees_list, new_taxon_idx, idx, s_mat_step, T):

        new_internal_idx = self.n_taxa + new_taxon_idx - 2


        # singleton {k}
        subtree_mat[s_mat_step, new_taxon_idx] = 1
        subtree_idx_mat[new_internal_idx, new_taxon_idx] = s_mat_step

        # {k} complementary
        subtree_mat[s_mat_step + 1, :new_taxon_idx] = 1
        subtree_mat[s_mat_step + 1, self.n_taxa: new_internal_idx + 1] = 1
        subtree_idx_mat[new_taxon_idx, new_internal_idx] = s_mat_step + 1


        # update T i->j j->i distance *********************

        # K = T.clone()

        ij = subtree_mat[subtree_idx_mat[idx[0], idx[1]]]
        ji = subtree_mat[subtree_idx_mat[idx[1], idx[0]]]

        T[ij==1] += ji
        T[ji == 1] += ij

        T[new_taxon_idx] = T[idx[0]] * ij + T[idx[1]] * ji


        # update subtrees_dist sub i <-> sub j
        # print(subtrees_dist_mat)


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
        subtree_idx_mat[new_internal_idx, idx[1]] = s_mat_step + 2

        # j -> i to k -> i
        subtree_mat[s_mat_step + 3] = ji
        subtree_idx_mat[new_internal_idx, idx[0]] = s_mat_step + 3

        # add k and new internal to previous
        subtree_mat[:s_mat_step, new_taxon_idx] = subtree_mat[:s_mat_step, new_internal_idx] = \
            subtree_mat[:s_mat_step, idx[0]] + subtree_mat[:s_mat_step, idx[1]] - \
            (subtree_mat[:s_mat_step, idx[0]] * subtree_mat[:s_mat_step, idx[1]])


        # adjust idxs
        subtree_idx_mat[idx[0], new_internal_idx] = subtree_idx_mat[idx[0], idx[1]]
        subtree_idx_mat[idx[1], new_internal_idx] = subtree_idx_mat[idx[1], idx[0]]
        subtree_idx_mat[idx[0], idx[1]] = subtree_idx_mat[idx[1], idx[0]] = 0

        # ff = self.powers[T[new_taxon_idx, :new_taxon_idx]]
        # k_leaves_dist = (self.d[new_taxon_idx, :new_taxon_idx] * self.powers[T[new_taxon_idx, :new_taxon_idx]]).unsqueeze(-1)
        # ll = subtree_mat[:s_mat_step + 3, :new_taxon_idx].to(torch.float64)
        # oo = torch.matmul(ll, k_leaves_dist)
        # kk = torch.matmul(ll, k_leaves_dist).squeeze()
        sij = subtree_mat[:s_mat_step + 4, idx[0]] * subtree_mat[:s_mat_step + 4, idx[1]] == 1
        si = torch.matmul(subtree_mat[:s_mat_step + 4], subtree_mat[subtree_idx_mat[new_internal_idx, idx[0]]]) > 0
        si = si * (~sij)
        sj = (~ si) * (~sij)
        # sik = si * (~( si * subtree_mat[:s_mat_step + 4, new_taxon_idx] == 1))
        # sjk = sj * (~( sj * subtree_mat[:s_mat_step + 4, new_taxon_idx] == 1))
        # si = si * (~ sik)
        # sj = sj * (~ sjk)
        self.mask[:s_mat_step + 4] = si
        mask2 = sj.repeat((si.sum(), 1))
        subtrees_dist_mat[self.mask, :s_mat_step + 4] *= 1 - mask2 / 2
        subtrees_dist_mat[:s_mat_step + 4, self.mask] *= 1 - mask2.T / 2


        # print(subtrees_dist_mat)
        subtrees_dist_mat[subtree_idx_mat[new_internal_idx, new_taxon_idx], :new_taxon_idx] = \
            subtrees_dist_mat[:new_taxon_idx, subtree_idx_mat[new_internal_idx, new_taxon_idx]] = \
            torch.matmul(subtree_mat[:new_taxon_idx, :new_taxon_idx].to(torch.float64),
                         (self.d[new_taxon_idx, :new_taxon_idx] * self.powers[T[new_taxon_idx, :new_taxon_idx]])
                         .unsqueeze(-1)).squeeze()
        # print(subtrees_dist_mat)



        return subtree_mat, subtree_idx_mat, T, subtrees_dist_mat


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


torch.set_printoptions(precision=2, linewidth=150)

random.seed(0)
np.random.seed(0)

n = 4

d = np.random.uniform(0,1,(n, n))
d = np.triu(d) + np.triu(d).T
np.fill_diagonal(d, 0)

device = 'cpu'
# device = 'cuda:0'

t = time.time()
model = PrecomputeTorch3(d, device=device)

model.solve()
print(time.time() - t)

print(model.subtree_dist*10)
print(model.check_dist()*10)
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




