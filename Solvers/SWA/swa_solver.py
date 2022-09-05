import copy

import numpy as np

from Solvers.solver import Solver


class swa_solver(Solver):

    def __init__(self, d):
        super(swa_solver, self).__init__(d)

    def solve(self):
        adj_mat = self.initial_mat()
        for i in range(self.n):
            idxs_list = np.nonzero(adj_mat[:, self.n:])
            idxs_list[1] += self.n
            min_val = 10 ** 5
            for idxs in idxs_list:
                sol = self.add_node(copy.deepcopy(adj_mat[:3 + i, :3 + i]), self.d[3 + i])  # to correct
