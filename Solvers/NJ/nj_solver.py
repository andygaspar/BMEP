import copy

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from Data_.Datasets.bmep_dataset import BMEP_Dataset
from Solvers.solver import Solver


class NjSolver(Solver):

    def __init__(self, d):
        super().__init__(d)
        self.solution_assignment = None
        self.max_val = 1e5
        self.d_update = np.ones_like(self.d) * self.max_val
        self.d_update[:self.n, :self.n] = d[:self.n, :self.n]
        self.q = np.ones_like(self.d_update) * self.max_val
        self.adj = None

        self.taxa = dict(zip(range(self.n), range(self.n)))

    def solve(self):
        sol = []
        idx_list = [i for i in range(self.n)]
        u = None
        for step in range(1, self.n - 2):
            self.compute_q(idx_list, step, u)
            u = np.unravel_index(np.argmin(self.q), self.q.shape)
            for el in u:
                idx_list.remove(el)
            idx_list.append(self.n + step)
            sol.append(u)
            self.update_d(idx_list, step, u)

        adj_mat = np.zeros((self.m, self.m), dtype=int)
        adj_mat[:self.n, self.n] = adj_mat[self.n, :self.n] = 1
        sol = self.solve_()
        for step, s in enumerate(sol):
            i, j = s
            k = np.nonzero(adj_mat[i, self.n:])[0] + self.n
            adj_mat[i, k] = adj_mat[k, i] = adj_mat[j, k] = adj_mat[k, j] = 0  # detach from previous link
            adj_mat[i, self.n + step + 1] = adj_mat[self.n + step + 1, i] = 1  # attach to new node
            adj_mat[j, self.n + step + 1] = adj_mat[self.n + step + 1, j] = 1  # attach to new node
            adj_mat[k, self.n + step + 1] = adj_mat[self.n + step + 1, k] = 1  # attach new node to prev link

            # g = nx.from_numpy_matrix(adj_mat)
            # nx.draw(g, with_labels=True)
            # plt.show()

        self.solution_assignment = sol
        self.solution = adj_mat
        self.obj_val = self.compute_obj_val_from_adj_mat(adj_mat, self.d, self.n)

        # s = self.solve_()
        # print(self.solution_assignment)
        # print(s, '\n\n')

    def solve_(self):
        sol = []
        d = copy.deepcopy(self.d[:self.n, :self.n])
        for step in range(1, self.n - 2):
            q = self.compute_q_(d)
            u = np.unravel_index(np.argmin(q), q.shape)
            sol.append((self.taxa[u[0]], self.taxa[u[1]]))
            d = self.update_d_(d, u)
            prev_vals = self.taxa.values()
            self.taxa = dict(zip(range(d.shape[0]), [self.n + step] + [i for i in prev_vals if i not in u]))

        return sol

    def compute_q(self, idx_list, step, u=None):
        if u is not None:
            f, g = u
            self.q[f] = self.q[:, f] = self.q[g] = self.q[:, g] = self.max_val

        n = self.n - step
        for i in idx_list:
            for j in idx_list[idx_list.index(i) + 1:]:
                self.q[i, j] = (n - 1) * self.d_update[i, j] - sum(self.d_update[i, idx_list]) \
                               - sum(self.d_update[j, idx_list])

    def update_d(self, idx_list, step, u):
        f, g = u
        for i in idx_list:
            self.d_update[i, self.n + step] = self.d_update[self.n + step, i] = \
                (self.d_update[f, i] + self.d_update[g, i] - self.d_update[f, g])/2

    @staticmethod
    def compute_q_(d):
        n = d.shape[0]
        q = np.zeros_like(d)
        for i in range(n):
            for j in range(i + 1, n):
                q[i, j] = (n - 2) * d[i, j] - sum(d[i]) - sum(d[j])
        return q

    def update_d_(self, d, u):
        n = d.shape[0]
        f, g = u
        new_d = np.zeros((n-1, n-1))
        idxs = [k for k in range(n) if k not in u]
        new_d[0, 1:] = new_d[1:, 0] = np.array([d[f, k] + d[g, k] - d[f, g] for k in idxs])/2
        new_d[1:, 1:] = d[idxs][:, idxs]
        return new_d


