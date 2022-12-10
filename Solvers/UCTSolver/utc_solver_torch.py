import numpy as np
import torch

from Solvers.UCTSolver.node_torch import NodeTorch
from Solvers.UCTSolver.utils.utc_utils import run_nni_search
from Solvers.solver import Solver


class UtcSolverTorch(Solver):

    def __init__(self, d: np.array, rollout_, compute_scores, c_initial=2 ** (1 / 2)):
        super(UtcSolverTorch, self).__init__(d)
        self.numpy_d = self.d
        self.d = torch.Tensor(self.d).to(self.device)

        self.rollout_ = rollout_
        self.compute_scores = compute_scores
        self.adj_mat_sparse = None
        self.root = None
        self.init_c = c_initial
        self.n_nodes = 0

    def solve(self, n_iterations=100):
        adj_mat = self.initial_adj_mat(device=self.device, n_problems=1)
        self.root = NodeTorch(adj_mat, step_i=3, d=self.d, n_taxa=self.n_taxa, c=self.init_c, parent=None,
                                rollout_=self.rollout_, compute_scores=self.compute_scores, device=self.device)
        self.obj_val, self.solution = self.root.expand()

        for iteration in range(n_iterations):
            node = self.root
            while not node.is_terminal() and node.is_expanded():
                node = node.best_child()

            if node.is_terminal():
                break
            run_best = node.expand()
            if run_best[0] < self.obj_val:
                self.obj_val, self.solution = run_best

        improved, best_val, best_solution = \
            run_nni_search(self.solution, self.obj_val, self.d, self.n_taxa, self.m)
        if improved:
            self.solution = best_solution
        self.obj_val = self.obj_val.item()
        self.T = self.get_tau(self.solution.to('cpu').numpy()).astype(np.int8)[:self.n_taxa, :self.n_taxa]
        self.d = self.numpy_d
        self.obj_val = self.compute_obj()
        self.adj_mat_sparse = torch.nonzero(torch.triu(self.solution))

        self.count_nodes()
        print(self.n_nodes)

    def recursion_counter(self, node):
        if node._children is None:
            return 1
        counter = 0
        for child in node._children:
            counter += self.recursion_counter(child)
        return counter

    def count_nodes(self):
        self.n_nodes = self.recursion_counter(self.root)

