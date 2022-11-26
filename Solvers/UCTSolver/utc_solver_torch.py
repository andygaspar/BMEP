import numpy as np
import torch

from Solvers.SWA.swa_solver import SwaSolver
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.UCTSolver.node import Node
from Solvers.UCTSolver.node_torch import NodeTorch
from Solvers.solver import Solver


class UtcSolverTorch(Solver):

    def __init__(self, d: np.array):
        super(UtcSolverTorch, self).__init__(d)
        self.d = torch.Tensor(self.d).to(self.device)

    def solve(self, n_iterations=100, parallel=True):
        adj_mat = self.initial_adj_mat(device=self.device, n_problems=1)
        root = NodeTorch(self.d, adj_mat, self.n_taxa, device=self.device)
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

        self.obj_val = self.obj_val.item()
        self.T = self.get_tau_(self.solution, self.n_taxa)
