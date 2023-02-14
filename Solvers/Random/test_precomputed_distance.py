import copy
import random
import time

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from Data_.Datasets.bmep_dataset import BMEP_Dataset
from Solvers.solver import Solver


class Precompute(Solver):

    def __init__(self, d, sorted_d=False):
        super(Precompute, self).__init__(d, sorted_d)
        self.subtrees_mat = None

    def solve(self, start=3, adj_mat=None):
        adj_mat = self.initial_adj_mat() if adj_mat is None else adj_mat
        subtrees_mat = self.initial_sub_tree_mat()
        T = self.init_tau()
        for i in range(start, self.n_taxa):
            idxs_list = np.array(np.nonzero(np.triu(adj_mat))).T
            idxs = random.choice(idxs_list)
            # print(adj_mat)
            adj_mat = self.add_node(adj_mat, idxs, i, self.n_taxa)
            self.update_tau(T, subtrees_mat, idxs, i)
            subtrees_mat = self.add_subtrees(subtrees_mat, i, idxs)

        self.subtrees_mat = subtrees_mat
        self.solution = adj_mat
        self.obj_val = self.compute_obj_val_from_adj_mat(adj_mat, self.d, self.n_taxa)
        print(T[:self.n_taxa, :self.n_taxa])
        self.T = self.get_tau(self.solution)

    def initial_sub_tree_mat(self):
        subtree_mat = np.zeros((self.m, self.m, self.m))

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

        node_on_side = np.nonzero(subtree_mat[:, :, idx[0]])

        # self.print_sub_mat(adj, subtree_mat)
        # print('\n')

        # move i->j to new_taxon_idx->j and j->i to new_taxon_idx->i
        subtree_mat[new_internal_idx, idx[1]] = subtree_mat[idx[0], idx[1]]
        subtree_mat[new_internal_idx, idx[0]] = subtree_mat[idx[1], idx[0]]

        subtree_mat[idx[0], new_internal_idx] = subtree_mat[idx[0], idx[1]]
        subtree_mat[idx[1], new_internal_idx] = subtree_mat[idx[1], idx[0]]

        subtree_mat[idx[0], new_internal_idx, new_internal_idx] = \
            subtree_mat[idx[0], new_internal_idx, new_taxon_idx] = 1
        subtree_mat[idx[1], new_internal_idx, new_internal_idx] = \
            subtree_mat[idx[1], new_internal_idx, new_taxon_idx] =  1

        subtree_mat[idx[0], idx[1]] = subtree_mat[idx[1], idx[0]] = 0

        # self.print_sub_mat(adj, subtree_mat)
        # print('\n')

        subtree_mat[new_taxon_idx, new_internal_idx, :new_taxon_idx] = 1
        subtree_mat[new_taxon_idx, new_internal_idx, self.n_taxa : new_internal_idx + 1] = 1
        subtree_mat[new_internal_idx , new_taxon_idx, new_taxon_idx] = 1

        # self.print_sub_mat(adj, subtree_mat)
        # print('\n')

        subtree_mat[node_on_side[0], node_on_side[1], [new_taxon_idx for _ in range(len(node_on_side[0]))]] = 1
        subtree_mat[node_on_side[0], node_on_side[1], [new_internal_idx for _ in range(len(node_on_side[0]))]] = 1

        # self.print_sub_mat(adj, subtree_mat)

        return subtree_mat

    def update_tau(self, T, subtrees_mat, idx, new_taxon):

        A = np.nonzero(subtrees_mat[idx[0], idx[1]])[0]
        B = np.nonzero(subtrees_mat[idx[1], idx[0]])[0]

        internal_node = new_taxon + self.n_taxa - 2

        T[internal_node, new_taxon] = T[new_taxon, internal_node] = 1

        T[[new_taxon for _ in range(len(A))], A] = T[[idx[0] for _ in range(len(A))], A] + 1
        T[[internal_node for _ in range(len(A))], A] = T[[idx[0] for _ in range(len(A))], A]

        T[A, [new_taxon for _ in range(len(A))]] = T[A, [idx[0] for _ in range(len(A))]] + 1
        T[A, [internal_node for _ in range(len(A))]] = T[A, [idx[0] for _ in range(len(A))]]

        T[[new_taxon for _ in range(len(B))], B] = T[[idx[1] for _ in range(len(B))], B] + 1
        T[[internal_node for _ in range(len(B))], B] = T[[idx[1] for _ in range(len(B))], B]

        T[B, [new_taxon for _ in range(len(B))]] = T[B, [idx[1] for _ in range(len(B))]] + 1
        T[B, [internal_node for _ in range(len(B))]] = T[B, [idx[1] for _ in range(len(B))]]

        T[A, B] += 1
        T[B, A] += 1

        return T
    def init_sub_dist(self):
        subtree_dist = np.zeros((self.m, self.m))
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
        T = np.zeros((self.m, self.m))
        T[0, 1] = T[0, 2] = T[1, 0] = T[1, 2] = T[2, 0] = T[2, 1] = 2
        T[0, self.n_taxa] = T[1, self.n_taxa] = T[2, self.n_taxa] =  \
            T[self.n_taxa, 0] = T[self.n_taxa, 1] = T[self.n_taxa, 2] = 1

        return T


random.seed(0)
np.random.seed(0)

n = 10

d = np.random.uniform(0,1,(n, n))
d = np.triu(d) + np.triu(d).T

t = time.time()
model = Precompute(d)
model.solve()
print(time.time() - t)
print(model.T)

