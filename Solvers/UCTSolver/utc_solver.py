import numpy as np

from Solvers.SWA.swa_solver import SwaSolver
from Solvers.UCTSolver.node import Node
from Solvers.solver import Solver


class UtcSolver(Solver):

    def __init__(self, d: np.array):
        super(UtcSolver, self).__init__(d)

    def swa_policy(self, adj_mat: np.array, step: int):
        swa_ = SwaSolver(self.d, sorted_d=True)
        swa_.solve(start=step, adj_mat=adj_mat)
        return swa_.obj_val, swa_.solution

    def solve(self, n_iterations=100):
        adj_mat = self.initial_adj_mat()
        root = Node(adj_mat, self.n_taxa, self.add_node, self.swa_policy)
        self.obj_val, self.solution = root.expand()

        for iter in range(n_iterations):
            node = root
            while not node.is_terminal() and node.is_expanded():
                node = node.best_child()

            if node.is_terminal():
                break
            run_best = node.expand()
            if run_best[0] < self.obj_val:
                self.obj_val, self.solution = run_best

