import copy

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from Data_.Datasets.bmep_dataset import BMEP_Dataset
from Solvers.solver import Solver


class SwaSolverNew(Solver):

    def __init__(self, d, sorted_d=False):
        super(SwaSolverNew, self).__init__(d, sorted_d)
        self.obj_val_ = None

    def solve(self, start=3, adj_mat=None):
        adj_mat = self.initial_adj_mat() if adj_mat is None else adj_mat
        adj = np.array([self.n_taxa]*3 + [0]*(self.m - 3))
        min_val, min_adj_mat = None, None
        min_val_, min_adj = None, None
        for i in range(start, self.n_taxa):
            min_val = 10 ** 5
            min_val_ = 10 ** 5
            minor_idxs = [j for j in range(i + 1)] + [j for j in range(self.n_taxa, self.n_taxa + i - 1)]
            idxs_list = np.array(np.nonzero(np.triu(adj_mat))).T
            m = (i + 1) * 2 - 2
            T = np.empty((m, m), dtype=int)
            for selection, idxs in enumerate(idxs_list):
                sol = self.add_node(copy.deepcopy(adj_mat), idxs, i, self.n_taxa)
                obj_val = self.compute_obj_val_from_adj_mat(sol[minor_idxs][:, minor_idxs], self.d[:i+1, :i+1], i+1)

                new_adj = self.add_node_(adj, selection, i, self.n_taxa)
                obj_val_ = self.compute_obj_(new_adj, i + 1, T, m)
                if obj_val < min_val:
                    min_val, min_adj_mat = obj_val, sol

                if obj_val_ < min_val_:
                    min_val_, min_adj = obj_val, new_adj

            adj_mat = min_adj_mat
            adj = min_adj

        self.solution = adj_mat
        self.obj_val = min_val
        self.obj_val_ = min_val_
        P = 1

    @staticmethod
    def add_node_(adj, selection, i, n_taxa):
        new_adj = copy.copy(adj)
        new_adj[i] = i + n_taxa - 2
        new_adj[selection] = new_adj[adj[selection]] = i + n_taxa - 2
        new_adj[i + n_taxa - 2] = adj[selection]

        return new_adj

    def compute_obj_(self, adj, step, T, m):
        T.fill(step)
        adj_reduced = adj[adj > 0] - step + 2
        elements = range(m)
        T[elements, adj_reduced] = T[adj_reduced, elements] = 1
        np.fill_diagonal(T, 0)  # diagonal elements should be zero
        for i in range(m):
            # The second term has the same shape as A due to broadcasting
            T = np.minimum(T, T[i, :][np.newaxis, :] + T[:, i][:, np.newaxis])
        T = T[:step, :step]
        r = range(step)
        return np.sum([self.d[i, j] / self.powers[T[i, j]] for i in r for j in r if i < j]) * 2





