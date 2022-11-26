import numpy as np
import torch

from Solvers.SWA.swa_solver import SwaSolver
from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.UCTSolver.node import Node
from Solvers.solver import Solver


class UtcSolver(Solver):

    def __init__(self, d: np.array, policy=SwaSolver):
        super(UtcSolver, self).__init__(d)
        self.policy = policy
        self.to_tensor = True if policy == SwaSolverTorch else False
        self.adj_mat_sparse = None

    def run_policy(self, adj_mat: np.array, step: int):
        solver = self.policy(self.d, sorted_d=True)
        solver.solve(start=step, adj_mat=adj_mat)
        return solver.obj_val, solver.solution

    def solve(self, n_iterations=100):
        adj_mat = self.initial_adj_mat().astype(np.int8) if not self.to_tensor else \
            self.initial_adj_mat(device=self.device, n_problems=1)
        add_node = self.add_node if not self.to_tensor else SwaSolverTorch.add_nodes
        root = Node(adj_mat, self.n_taxa, add_node, self.run_policy, to_tensor=self.to_tensor)
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

        self.T = self.get_tau_(self.solution, self.n_taxa)
        self.obj_val = self.compute_obj()
        if type(self.solution) != torch.tensor:
            solution = torch.tensor(self.solution)
            self.adj_mat_sparse = torch.nonzero(torch.triu(solution))
        else:
            self.adj_mat_sparse = torch.nonzero(torch.triu(self.solution))
