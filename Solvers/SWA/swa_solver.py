import copy

import numpy as np

from Solvers.solver import Solver


class SwaSolver(Solver):

    def __init__(self, d):
        super(SwaSolver, self).__init__(d)

    def solve(self):
        adj_mat = self.initial_mat()
        min_val = 10 ** 5
        for i in range(3, self.n):
            min_val = 10 ** 5
            minor_idxs = [j for j in range(i + 1)] + [j for j in range(self.n, self.n + i - 1)]
            idxs_list = np.array(np.nonzero(np.triu(adj_mat))).T
            # idxs_list[:, 1] += i
            for idxs in idxs_list:
                sol = self.add_node(copy.deepcopy(adj_mat), idxs, i, self.n)
                obj_val = self.compute_obj_val_from_adj_mat(sol[minor_idxs][:, minor_idxs], self.d[:i+1, :i+1], i+1)
                if obj_val < min_val:
                    min_val, adj_mat = obj_val, sol
        self.solution = adj_mat
        self.obj_val = min_val
