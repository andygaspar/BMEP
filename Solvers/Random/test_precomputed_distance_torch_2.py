import random
import time
import numpy as np
import torch

from Solvers.Random.test_precomputed_distance_torch import PrecomputeTorch
from Solvers.solver import Solver


class PrecomputeTorch2(Solver):

    def __init__(self, d, sorted_d=False):
        super(PrecomputeTorch2, self).__init__(d, sorted_d)
        self.device = 'cpu'
        # self.d = torch.tensor(self.d, device=self.device)
        # self.powers = self.powers.to('cpu')
        self.subtrees_mat = None
        self.subtree_dist = None
        self.T_new = None

    def solve(self, start=3, adj_mat=None):
        t = time.time()
        adj_mat = self.initial_adj_mat(self.device) if adj_mat is None else adj_mat
        subtrees_mat, subtrees_idx_mat = self.initial_sub_tree_mat()
        T = self.init_tau()
        s_mat_step = 6
        # subtree_dist = self.init_sub_dist()
        for step in range(start, self.n_taxa):
            choices = 3 + (step - 3) * 2
            idxs_list = torch.nonzero(torch.triu(adj_mat))
            rand_idxs = random.choices(range(choices))
            idxs_list = idxs_list[rand_idxs][0]
            adj_mat = self.add_node(adj_mat, idxs_list, step, self.n_taxa)
            # self.update_tau(T, subtrees_mat, idxs_list, step)
            subtrees_mat, subtrees_idx_mat = \
                self.add_subtrees(subtrees_mat, subtrees_idx_mat, step, idxs_list, s_mat_step)
            s_mat_step += 4

        self.subtrees_mat = subtrees_mat
        self.solution = adj_mat
        # self.obj_val = self.compute_obj_val_from_adj_mat(adj_mat, self.d, self.n_taxa)
        # print(T[:self.n_taxa, :self.n_taxa])
        # self.T = self.get_tau(self.solution)
        self.T = T
        # self.device = 'cuda:0'
        print('build time', time.time() - t)
        # t = time.time()
        # self.subtree_dist = self.compute_dist()
        # print(time.time() - t)


    def initial_sub_tree_mat(self):
        subtree_mat = torch.zeros((2*(2*self.n_taxa - 3), 2*(2*self.n_taxa - 3)), device=self.device, dtype=torch.float)
        subtree_idx_mat = torch.zeros((self.m, self.m), device=self.device, dtype=torch.long)

        # 0
        subtree_mat[0, 0]  = 1
        subtree_mat[1, 1]  = 1
        subtree_mat[2, 2]  = 1

        subtree_idx_mat[self.n_taxa, 0] = 0
        subtree_idx_mat[self.n_taxa, 1] = 1
        subtree_idx_mat[self.n_taxa, 2] = 2

        # 1
        subtree_mat[3, self.n_taxa] = subtree_mat[3, 1] = subtree_mat[3, 2] = 1
        subtree_mat[4, self.n_taxa] = subtree_mat[4, 0] = subtree_mat[4, 2] = 1
        subtree_mat[5, self.n_taxa] = subtree_mat[5, 0] = subtree_mat[5, 1] = 1

        # 1
        subtree_idx_mat[0, self.n_taxa] = 3
        subtree_idx_mat[1, self.n_taxa] = 4
        subtree_idx_mat[2, self.n_taxa] = 5

        return  subtree_mat, subtree_idx_mat

    def add_subtrees(self, subtree_mat, subtree_idx_mat, new_taxon_idx, idx, s_mat_step):

        new_internal_idx = self.n_taxa + new_taxon_idx - 2

        # singleton {k}
        subtree_mat[s_mat_step, new_taxon_idx] = 1
        subtree_idx_mat[new_internal_idx, new_taxon_idx] = s_mat_step

        # {k} complementary
        subtree_mat[s_mat_step + 1] = 1
        subtree_mat[s_mat_step + 1, new_taxon_idx] = 0
        subtree_idx_mat[new_taxon_idx, new_internal_idx] = s_mat_step + 1

        # i -> j to k -> j
        subtree_mat[s_mat_step + 2] = subtree_mat[subtree_idx_mat[idx[0], idx[1]]]
        subtree_idx_mat[new_internal_idx, idx[1]] = s_mat_step + 2

        # j -> i to k -> i
        subtree_mat[s_mat_step + 3] = subtree_mat[subtree_idx_mat[idx[1], idx[0]]]
        subtree_idx_mat[new_internal_idx, idx[0]] = s_mat_step + 3

        # add k to previous

        subtree_mat[:s_mat_step, new_taxon_idx] = \
            subtree_mat[:s_mat_step, idx[0]] + subtree_mat[:s_mat_step, idx[1]] - \
            (subtree_mat[:s_mat_step, idx[0]] * subtree_mat[:s_mat_step, idx[1]])

        # adjust idxs
        subtree_idx_mat[idx[0], new_internal_idx] = subtree_idx_mat[idx[0], idx[1]]
        subtree_idx_mat[idx[1], new_internal_idx] = subtree_idx_mat[idx[1], idx[0]]
        subtree_idx_mat[idx[0], idx[1]] = subtree_idx_mat[idx[1], idx[0]] = 0

        return subtree_mat, subtree_idx_mat

    def update_tau(self, T, subtrees_mat, idx, new_taxon):

        A = torch.nonzero(subtrees_mat[idx[0], idx[1]]).T.squeeze(0)
        B = torch.nonzero(subtrees_mat[idx[1], idx[0]]).T.squeeze(0)

        internal_node = new_taxon + self.n_taxa - 2

        T[internal_node, new_taxon] = T[new_taxon, internal_node] = 1
        n_A, i_A = [new_taxon for _ in range(len(A))], [internal_node for _ in range(len(A))]
        idx_A = [idx[0] for _ in range(len(A))]

        T[n_A, A] = T[idx_A, A] + 1
        T[i_A, A] = T[idx_A, A]

        T[A, n_A] = T[A, idx_A] + 1
        T[A, i_A] = T[A, idx_A]

        n_B, i_B = [new_taxon for _ in range(len(B))], [internal_node for _ in range(len(B))]
        idx_B = [idx[1] for _ in range(len(B))]

        T[n_B, B] = T[idx_B, B] + 1
        T[i_B, B] = T[idx_B, B]

        T[B, n_B] = T[B, idx_B] + 1
        T[B, i_B] = T[B, idx_B]

        idxs = torch.cartesian_prod(A, B)

        T[idxs[:, 0], idxs[:, 1]] += 1
        T[idxs[:, 1], idxs[:, 0]] += 1

        return T
    def init_sub_dist(self):
        subtree_dist = torch.zeros((self.m, self.m), device=self.device, dtype=torch.bool)
        subtree_dist[:self.n_taxa, :self.n_taxa] = self.d

        return subtree_dist

    @staticmethod
    def print_sub_mat(adj, sub):
        for i in range(10):
            for j in range(10):
                if adj[i, j] > 0.5:
                    print(i, j, sub[i, j])

    def swap(self, idxs):
        pass

    def init_tau(self):
        T = torch.zeros((self.m, self.m), dtype=torch.long, device=self.device)
        T[0, 1] = T[0, 2] = T[1, 0] = T[1, 2] = T[2, 0] = T[2, 1] = 2
        T[0, self.n_taxa] = T[1, self.n_taxa] = T[2, self.n_taxa] =  \
            T[self.n_taxa, 0] = T[self.n_taxa, 1] = T[self.n_taxa, 2] = 1

        return T

    def compute_dist(self):
        dist_mat = torch.zeros((2*(2*self.n_taxa - 3), 2*(2*self.n_taxa - 3)), dtype=torch.float64, device=self.device)
        d = self.d * self.powers[self.T[:self.n_taxa, :self.n_taxa]]

        edges = torch.nonzero(self.solution)
        idx_dict = dict(zip([tuple(edge.tolist()) for edge in edges], range(edges.shape[0])))
        for edge in edges:
            for o_edge in edges:
                if (self.subtrees_mat[edge[0], edge[1], :] * self.subtrees_mat[o_edge[0], o_edge[1], :]).sum() == 0:
                    A = torch.nonzero(self.subtrees_mat[edge[0], edge[1], :self.n_taxa], as_tuple=True)[0]
                    B = torch.nonzero(self.subtrees_mat[o_edge[0], o_edge[1], :self.n_taxa], as_tuple=True)[0]
                    idxs = torch.cartesian_prod(A, B)
                    dist_mat[idx_dict[(edge[0].item(), edge[1].item())], idx_dict[(o_edge[0].item(), o_edge[1].item())]] = (d[idxs[:, 0], idxs[:, 1]] ).sum()

        dist_mat = dist_mat + dist_mat.T

        return dist_mat

    def compute_dist_1(self):
        dist_mat = torch.zeros((2*(2*self.n_taxa - 3), 2*(2*self.n_taxa - 3)), dtype=torch.float64, device=self.device)
        d = self.d * self.powers[self.T[:self.n_taxa, :self.n_taxa]]
        edges = torch.nonzero(self.solution)
        idx_dict = dict(zip([tuple(edge.tolist()) for edge in edges], range(edges.shape[0])))
        s_mat = self.subtrees_mat[edges[:, 0], edges[:, 1], :].clone()

        for edge in edges:
            for o_edge in edges:
                if (self.subtrees_mat[edge[0], edge[1], :] * self.subtrees_mat[o_edge[0], o_edge[1], :]).sum() == 0:
                    idx_1 = idx_dict[(edge[0].item(), edge[1].item())]
                    idx_2 = idx_dict[(o_edge[0].item(), o_edge[1].item())]
                    A = torch.nonzero(s_mat[idx_1, :self.n_taxa], as_tuple=True)[0]
                    B = torch.nonzero(s_mat[idx_2, :self.n_taxa], as_tuple=True)[0]
                    idxs = torch.cartesian_prod(A, B)
                    dist_mat[idx_1, idx_2] = (d[idxs[:, 0], idxs[:, 1]]).sum()

        dist_mat = dist_mat + dist_mat.T

        return dist_mat

    def compute_dist_2(self):
        # self.d = self.d.to(self.device)
        # self.powers = self.powers.to(self.device)
        # self.subtrees_mat = self.subtrees_mat.to(self.device)
        # self.T = self.T.to(self.device)
        d = self.d * self.powers[self.T[:self.n_taxa, :self.n_taxa]]
        edges = torch.nonzero(self.solution)
        # idx_dict = dict(zip([tuple(edge.tolist()) for edge in edges], range(edges.shape[0])))
        s_mat = self.subtrees_mat[edges[:, 0], edges[:, 1], :].to(torch.float64)

        a = s_mat.repeat(s_mat.shape[0],1)
        b = s_mat.repeat_interleave(s_mat.shape[0], dim=0)
        res = (1 - (a*b).sum(dim=-1))
        res[res<0] = 0
        s = torch.matmul(s_mat[:, :self.n_taxa].flatten().unsqueeze(1), s_mat[:, :self.n_taxa].flatten().unsqueeze(0))
        dist = (d.repeat(s_mat.shape[0], s_mat.shape[0]) * s).split(self.n_taxa, dim=-1)
        dist = torch.stack(dist).view(-1, self.n_taxa, self.n_taxa).transpose(1,2)

        return (dist.reshape(-1, self.n_taxa**2).sum(dim=-1) * res).reshape((s_mat.shape[0], s_mat.shape[0]))


    def compute_dist_3(self):
        # self.d = self.d.to(self.device)
        # self.powers = self.powers.to(self.device)
        # self.subtrees_mat = self.subtrees_mat.to(self.device)
        # self.T = self.T.to(self.device)
        d = self.d * self.powers[self.T[:self.n_taxa, :self.n_taxa]]
        edges = torch.nonzero(self.solution)
        # idx_dict = dict(zip([tuple(edge.tolist()) for edge in edges], range(edges.shape[0])))
        s_mat = self.subtrees_mat[edges[:, 0], edges[:, 1], :].to(torch.float64)

        a = s_mat.repeat(s_mat.shape[0],1)
        b = s_mat.repeat_interleave(s_mat.shape[0], dim=0)
        res = (1 - (a*b).sum(dim=-1))
        res[res<0] = 0
        z = res.reshape((s_mat.shape[0], s_mat.shape[0]))
        s = torch.matmul(s_mat[:, :self.n_taxa].flatten().unsqueeze(1), s_mat[:, :self.n_taxa].flatten().unsqueeze(0))
        dist = (d.repeat(s_mat.shape[0], s_mat.shape[0]) * s).split(self.n_taxa, dim=-1)
        dist = torch.stack(dist).view(-1, self.n_taxa, self.n_taxa).transpose(1,2)

        return (dist.reshape(-1, self.n_taxa**2).sum(dim=-1) * res).reshape((s_mat.shape[0], s_mat.shape[0]))
torch.set_printoptions(linewidth=150)

random.seed(0)
np.random.seed(0)

n = 400

d = np.random.uniform(0,1,(n, n))
d = np.triu(d) + np.triu(d).T


t = time.time()
model = PrecomputeTorch(d)
model.solve()
print(time.time() - t)

t = time.time()
model = PrecomputeTorch2(d)
model.solve()
print(time.time() - t)



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
