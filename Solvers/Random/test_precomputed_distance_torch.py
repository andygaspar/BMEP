import copy
import random
import time

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

# from Data_.Datasets.bmep_dataset import BMEP_Dataset
from Solvers.solver import Solver


class PrecomputeTorch(Solver):

    def __init__(self, d, sorted_d=False):
        super(PrecomputeTorch, self).__init__(d, sorted_d)
        self.d = torch.tensor(self.d, device=self.device)
        self.subtrees_mat = None
        self.subtree_dist = None
        self.T_new = None

    def solve(self, start=3, adj_mat=None):
        t = time.time()
        adj_mat = self.initial_adj_mat(self.device) if adj_mat is None else adj_mat
        subtrees_mat = self.initial_sub_tree_mat()
        T = self.init_tau()
        # subtree_dist = self.init_sub_dist()
        for step in range(start, self.n_taxa):
            choices = 3 + (step - 3) * 2
            idxs_list = torch.nonzero(torch.triu(adj_mat))
            rand_idxs = random.choices(range(choices))
            idxs_list = idxs_list[rand_idxs][0]
            # print(adj_mat)
            adj_mat = self.add_node(adj_mat, idxs_list, step, self.n_taxa)
            self.update_tau(T, subtrees_mat, idxs_list, step)
            subtrees_mat = self.add_subtrees(subtrees_mat, step, idxs_list)

        self.subtrees_mat = subtrees_mat
        self.solution = adj_mat
        # self.obj_val = self.compute_obj_val_from_adj_mat(adj_mat, self.d, self.n_taxa)
        # print(T[:self.n_taxa, :self.n_taxa])
        # self.T = self.get_tau(self.solution)
        self.T = T
        print('build time', time.time() - t)
        self.subtree_dist = self.compute_dist()
        # self.T_new = T.numpy().astype(np.int32)[:self.n_taxa, :self.n_taxa]

        # print('equal', np.array_equal(self.T, self.T_new))

    def initial_sub_tree_mat(self):
        subtree_mat = torch.zeros((self.m, self.m, self.m), device=self.device, dtype=torch.bool)

        # 0
        subtree_mat[self.n_taxa, 0, 0]  = 1
        subtree_mat[self.n_taxa, 1, 1]  = 1
        subtree_mat[self.n_taxa, 2, 2]  = 1

        # 1
        subtree_mat[0, self.n_taxa, 1] = subtree_mat[0, self.n_taxa, 2] = subtree_mat[0, self.n_taxa, self.n_taxa] = 1
        subtree_mat[1, self.n_taxa, 0] = subtree_mat[1, self.n_taxa, 2] = subtree_mat[1, self.n_taxa, self.n_taxa] = 1
        subtree_mat[2, self.n_taxa, 0] = subtree_mat[2, self.n_taxa, 1] = subtree_mat[2, self.n_taxa, self.n_taxa] = 1

        return  subtree_mat

    def add_subtrees(self, subtree_mat, new_taxon_idx, idx):

        new_internal_idx = self.n_taxa + new_taxon_idx - 2

        node_on_side_i = torch.nonzero(subtree_mat[:, :, idx[0]])


        # move i->j to new_taxon_idx->j and j->i to new_taxon_idx->i
        subtree_mat[new_internal_idx, idx[1]] = subtree_mat[idx[0], idx[1]]
        subtree_mat[new_internal_idx, idx[0]] = subtree_mat[idx[1], idx[0]]

        subtree_mat[idx[0], new_internal_idx] = subtree_mat[idx[0], idx[1]]
        subtree_mat[idx[1], new_internal_idx] = subtree_mat[idx[1], idx[0]]

        subtree_mat[idx[0], new_internal_idx, new_internal_idx] = \
            subtree_mat[idx[0], new_internal_idx, new_taxon_idx] = 1
        subtree_mat[idx[1], new_internal_idx, new_internal_idx] = \
            subtree_mat[idx[1], new_internal_idx, new_taxon_idx] =  1



        subtree_mat[new_taxon_idx, new_internal_idx, :new_taxon_idx] = 1
        subtree_mat[new_taxon_idx, new_internal_idx, self.n_taxa : new_internal_idx + 1] = 1
        subtree_mat[new_internal_idx , new_taxon_idx, new_taxon_idx] = 1

        subtree_mat[node_on_side_i[:, 0], node_on_side_i[:, 1], [new_taxon_idx for _ in range(node_on_side_i.shape[0])]] = 1
        subtree_mat[node_on_side_i[:, 0], node_on_side_i[:, 1], [new_internal_idx for _ in range(node_on_side_i.shape[0])]] = 1

        subtree_mat[idx[0], idx[1]] = subtree_mat[idx[1], idx[0]] = 0
        return subtree_mat

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
        dist_mat = torch.zeros((self.m, self.m), device=self.device)
        # add pre-computation TO DO
        # d = self.d / 2** self.T[:self.n_taxa, :self.n_taxa]
        d = self.d * self.powers[self.T[:self.n_taxa, :self.n_taxa]]
        for i in range(self.m):
            for j in range(i + 1, self.m):
                a = torch.nonzero(self.subtrees_mat[i, :, j])[0][0]
                A = torch.nonzero(self.subtrees_mat[a, i, :self.n_taxa]).T.squeeze(0)
                b = torch.nonzero(self.subtrees_mat[j, :, i])[0][0]
                B = torch.nonzero(self.subtrees_mat[b, j, :self.n_taxa]).T.squeeze(0)
                idxs = torch.cartesian_prod(A, B)
                dist_mat[i, j] = (d[idxs[:, 0], idxs[:, 1]] ).sum()

        dist_mat = dist_mat + dist_mat.T

        return dist_mat


random.seed(0)
np.random.seed(0)

n = 300

d = np.random.uniform(0,1,(n, n))
d = np.triu(d) + np.triu(d).T

t = time.time()
model = PrecomputeTorch(d)
model.solve()
print(time.time() - t)


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

# TO DO comment
