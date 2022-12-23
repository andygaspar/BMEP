import numpy as np
import torch

from Solvers.UCTSolver.node_torch import NodeTorch
from Solvers.UCTSolver.utils.utc_utils import run_nni_search
from Solvers.UCTSolver.utils.utc_utils_batch import run_nni_search_batch_for_tracking
from Solvers.UCTSolver.utils.utils_rollout import swa_policy, swa_nni_policy
from Solvers.UCTSolver.utils.utils_scores import max_score_normalised
from Solvers.solver import Solver


class UtcSolverTorchBackTrack2(Solver):

    def __init__(self, d: np.array, rollout_, compute_scores, c_initial=2 ** (1 / 2), budget=1000):
        super(UtcSolverTorchBackTrack2, self).__init__(d)
        self.numpy_d = self.d

        self.rollout_ = rollout_
        self.compute_scores = compute_scores
        self.adj_mat_sparse = None
        self.root = None
        self.init_c = c_initial
        self.n_nodes = 0

        self.budget = budget

    def solve(self, n_iterations=100):
        # with torch.no_grad():
        self.d = torch.tensor(self.d, requires_grad=False).to(self.device)
        adj_mat = self.initial_adj_mat(device=self.device, n_problems=1)
        self.root = NodeTorch(adj_mat, step_i=3, d=self.d, n_taxa=self.n_taxa, c=self.init_c, parent=None,
                              rollout_=self.rollout_, compute_scores=self.compute_scores, powers=self.powers,
                              device=self.device)
        self.obj_val, self.solution, obj_vals, sol_adj_mat = self.root.expand_full()
        best_val, best_solution, trees, objs = run_nni_search_batch_for_tracking(sol_adj_mat, obj_vals, self.d,
                                                                               self.n_taxa,
                                                                               self.m, self.powers, self.device)
        b = torch.cat(objs)
        s = torch.argsort(b)
        b = b[s]
        sols = torch.cat(trees, dim=0)[s]

        for iteration in range(n_iterations):
            node = self.root
            while not node.is_terminal() and node.is_expanded():
                node = node.best_child()

            if node.is_terminal():
                break
            run_val, run_sol, obj_vals, sol_adj_mat = node.expand_full()
            best_val, best_solution, trees, obj_vals = run_nni_search_batch_for_tracking(sol_adj_mat, obj_vals, self.d,
                                                                                   self.n_taxa,
                                                                                   self.m, self.powers, self.device)
            # print(run_val, torch.min(mixed_run_val))

            # if run_val < self.obj_val * self.nni_tol:
            #     improved, nni_val, nni_sol = \
            #         run_nni_search(self.nni_iterations, run_sol, self.obj_val, self.d, self.n_taxa, self.m, self.device)
            #     if improved:
            #         run_val, run_sol = nni_val, nni_sol
            #         trajectory_id = self.tree_climb(run_sol)
            #         node_climbed = self.get_lower_node_in_trajectory(trajectory_id)
            #         node_climbed.update_and_backprop(run_val)

            if run_val < self.obj_val:
                self.obj_val, self.solution = run_val, run_sol



        self.obj_val = self.obj_val.item()
        self.T = self.get_tau(self.solution.to('cpu').numpy()).astype(np.int8)[:self.n_taxa, :self.n_taxa]
        self.d = self.numpy_d
        self.obj_val = self.compute_obj()
        self.adj_mat_sparse = torch.nonzero(torch.triu(self.solution))

    def recursion_counter(self, node):
        if node._children is None:
            return 1
        counter = 0
        for child in node._children:
            counter += self.recursion_counter(child)
        return counter

    def count_nodes(self):
        self.n_nodes = self.recursion_counter(self.root)

    def tree_climb(self, adj_mat):
        adj_mat = adj_mat.unsqueeze(0).repeat(3, 1, 1)
        last_inserted_taxa = self.n_taxa - 1

        # reorder matrix according to Pardi
        for step in range(self.m - 1, self.n_taxa, -1):

            if adj_mat[2][step, last_inserted_taxa] == 0:
                idx = torch.nonzero(adj_mat[2][last_inserted_taxa]).item()
                adj_mat = self.permute(adj_mat, step, idx)

            adj_mat[2][:, last_inserted_taxa] = adj_mat[2][last_inserted_taxa, :] = 0
            idxs = torch.nonzero(adj_mat[2][step])
            adj_mat[2][idxs[0], idxs[1]] = adj_mat[2][idxs[1], idxs[0]] = 1
            adj_mat[2][:, step] = adj_mat[2][step, :] = 0

            last_inserted_taxa -= 1

        # Trace insertion back
        id_ = ()
        last_inserted_taxa = self.n_taxa - 1
        for step in range(self.m - 1, self.n_taxa, -1):
            adj_mat[1][:, last_inserted_taxa] = adj_mat[1][last_inserted_taxa, :] = 0
            idxs = torch.nonzero(adj_mat[1][step])
            adj_mat[1][idxs[0], idxs[1]] = adj_mat[1][idxs[1], idxs[0]] = 1
            adj_mat[1][:, step] = adj_mat[1][step, :] = 0

            if idxs[0] < last_inserted_taxa:
                p = (int(idxs[0].item()),)
                id_ += p
            else:
                p = (int((torch.triu(adj_mat[1])[:idxs[0], :].sum() +
                          adj_mat[1][idxs[0], idxs[0]: idxs[1]].sum()).item()),)
                id_ += p

            last_inserted_taxa -= 1

        return tuple(reversed(id_))

    @staticmethod
    def permute(adj_mats, step, idx):
        adj_mats[:, step, :] += adj_mats[:, idx, :]
        adj_mats[:, idx, :] = adj_mats[:, step, :] - adj_mats[:, idx, :]
        adj_mats[:, step, :] -= adj_mats[:, idx, :]

        adj_mats[:, :, step] += adj_mats[:, :, idx]
        adj_mats[:, :, idx] = adj_mats[:, :, step] - adj_mats[:, :, idx]
        adj_mats[:, :, step] -= adj_mats[:, :, idx]
        return adj_mats

    def get_lower_node_in_trajectory(self, trajectory_id):
        node = self.root
        idx = 0
        while node.is_expanded():
            node = node._children[trajectory_id[idx]]
            idx += 1

        return node





