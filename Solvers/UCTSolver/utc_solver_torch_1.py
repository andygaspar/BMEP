import numpy as np
import torch

from Solvers.SWA.swa_solver import SwaSolver
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.UCTSolver.node import Node
from Solvers.UCTSolver.node_torch import NodeTorch
from Solvers.UCTSolver.node_torch_1 import NodeTorch_1
from Solvers.solver import Solver


class UtcSolverTorch_1(Solver):

    def __init__(self, d: np.array):
        super(UtcSolverTorch_1, self).__init__(d)
        self.numpy_d = self.d
        self.d = torch.Tensor(self.d).to(self.device)
        self.adj_mat_sparse = None
        self.root = None
        self.n_nodes = 0

    def solve(self, n_iterations=100):
        adj_mat = self.initial_adj_mat(device=self.device, n_problems=1)
        self.root = NodeTorch_1(self.d, adj_mat, self.n_taxa, device=self.device)
        self.obj_val, self.solution = self.root.expand()

        for iter in range(n_iterations):
            node = self.root
            while not node.is_terminal() and node.is_expanded():
                node = node.best_child()

            if node.is_terminal():
                break
            run_best = node.expand()
            if run_best[0] < self.obj_val:
                self.obj_val, self.solution = run_best

        self.obj_val = self.obj_val.item()
        self.T = self.get_tau(self.solution.to('cpu').numpy()).astype(np.int8)
        self.d = self.numpy_d
        self.obj_val = self.compute_obj()
        self.adj_mat_sparse = torch.nonzero(torch.triu(self.solution))

        self.count_nodes()

    def recursion_counter(self, node):
        if node._children is None:
            return 1
        counter = 0
        for child in node._children:
            counter += self.recursion_counter(child)
        return counter

    def count_nodes(self):
        self.n_nodes = self.recursion_counter(self.root)

