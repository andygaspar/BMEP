import copy

import networkx as nx
import numpy as np


class PardiSolver:

    def __init__(self, d):
        self.d = d
        self.n = self.d.shape[0]
        self.steps = 0
        self.best_val = 10**5
        self.solution = None

    def compute_obj_val(self, mat):
        g = nx.from_numpy_matrix(mat)
        Tau = nx.floyd_warshall_numpy(g)[:self.n, :self.n]
        return np.sum([self.d[i, j] / 2 ** (Tau[i, j]) for i in range(self.n) for j in range(self.n)])

    def recursion(self, mat, step=3):
        self.steps += 1
        selection = np.array(np.nonzero(np.triu(mat))).T
        recursion_step = selection.shape[0]
        if recursion_step == self.n + self.n-3:
            obj_val = self.compute_obj_val(mat)
            if obj_val < self.best_val:
                self.best_val = obj_val
                self.solution = mat
        else:
            for idxs in selection:
                new_mat = self.add_node(copy.deepcopy(mat), idxs, step)
                self.recursion(copy.deepcopy(new_mat), step + 1)

    def solve(self):
        mat = np.zeros((self.n + self.n-2, self.n + self.n-2))
        for i in range(3):
            mat[i, self.n] = mat[self.n, i] = 1
        self.recursion(copy.deepcopy(mat))

    def add_node(self, adj_mat, idxs, new_node_idx):
        adj_mat[idxs[0], idxs[1]] = adj_mat[idxs[1], idxs[0]] = 0  # detach selected
        adj_mat[idxs[0], self.n + new_node_idx - 2] = adj_mat[self.n + new_node_idx - 2, idxs[0]] = 1  # reattach selected to new
        adj_mat[idxs[1], self.n + new_node_idx - 2] = adj_mat[self.n + new_node_idx - 2, idxs[1]] = 1  # reattach selected to new
        adj_mat[new_node_idx, self.n + new_node_idx - 2] = adj_mat[self.n + new_node_idx - 2, new_node_idx] = 1  # attach new

        return adj_mat

