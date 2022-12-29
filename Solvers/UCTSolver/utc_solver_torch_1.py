import copy

import numpy as np
import torch

from Solvers.UCTSolver.node_torch import NodeTorch
from Solvers.UCTSolver.node_torch_backtrack import NodeTorchBackTrack
from Solvers.UCTSolver.utils.utc_utils import run_nni_search
from Solvers.UCTSolver.utils.utc_utils_batch import run_nni_search_batch_for_tracking
from Solvers.UCTSolver.utils.utils_rollout import swa_policy, swa_nni_policy, adjust_matrices
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

        self.expansion = 0

        self.budget = budget

    def solve(self, n_iterations=100):
        # with torch.no_grad():
        self.d = torch.tensor(self.d, requires_grad=False).to(self.device)
        adj_mat = self.initial_adj_mat(device=self.device, n_problems=1)
        self.root = NodeTorchBackTrack(step_i=3, d=self.d, n_taxa=self.n_taxa, c=self.init_c, parent=None,
                              rollout_=self.rollout_, compute_scores=self.compute_scores, powers=self.powers,
                              device=self.device)
        self.obj_val, self.solution, obj_vals, sol_adj_mat = self.root.expand(adj_mat)
        best_val, best_solution, trees, objs = run_nni_search_batch_for_tracking(sol_adj_mat, obj_vals, self.d,
                                                                               self.n_taxa,
                                                                               self.m, self.powers, self.device)

        self.expansion += obj_vals.shape[0]
        if best_val < self.obj_val:
            self.obj_val, self.solution = best_val, best_solution
        self.back_track(trees, objs)

        for iteration in range(n_iterations):
            ad_mat = copy.deepcopy(adj_mat)
            node = self.root
            step = 2
            while not node.is_terminal() and node.is_expanded():
                node = node.best_child()
                print(node.id)
                step += 1
                idxs = torch.nonzero(ad_mat)[node.id].unsqueeze(0).T
                idxs = tuple([idx for idx in idxs])
                ad_mat = self.add_nodes(ad_mat, idxs, step, self.n_taxa)

            if node.is_terminal():
                break

            run_val, run_sol, obj_vals, sol_adj_mat = node.expand(ad_mat)
            self.expansion += obj_vals.shape[0]
            best_val, best_solution, trees, objs = run_nni_search_batch_for_tracking(sol_adj_mat, obj_vals, self.d,
                                                                                   self.n_taxa,
                                                                                   self.m, self.powers, self.device)
            self.back_track(trees, objs)
            if best_val < run_val:
                run_val, run_sol = best_val, best_solution
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

    def back_track(self, trees, objs):
        objs = torch.cat(objs)
        idxs = torch.argsort(objs)
        objs = objs[idxs][:self.budget]
        print(objs[0].item())
        sols = torch.cat(trees, dim=0)[idxs][:self.budget]
        trajectories = self.tree_climb(sols)
        mean = trajectories.mean(dim=0, dtype=torch.float)
        var = trajectories.to(torch.float).var(dim=0)
        t_stat = torch.vstack([mean, var])
        print(t_stat)
        for i, tj in enumerate(trajectories):
            self.root.fill(tj,objs[i])

    def tree_climb(self, adj_mats):
        last_inserted_taxa = self.n_taxa - 1
        n_internals = self.m - self.n_taxa

        adj_mats = adjust_matrices(adj_mats, last_inserted_taxa, n_internals, self.n_taxa)
        adj_mats = adj_mats.unsqueeze(1).repeat(1, 2, 1, 1)
        reversed_idxs = torch.tensor([[i, i - 1] for i in range(1, adj_mats.shape[0]*2, 2)], device=self.device).flatten()
        trajectories = torch.zeros((adj_mats.shape[0], self.n_taxa - 3), dtype=torch.int)

        last_inserted_taxa = self.n_taxa - 1
        for step in range(self.m - 1, self.n_taxa, -1):

            adj_mats[:, 1, :, last_inserted_taxa] = adj_mats[:, 1, last_inserted_taxa, :] = 0
            idxs = torch.nonzero(adj_mats[:, 1, step])
            idxs = torch.column_stack([idxs, idxs[:, 1][reversed_idxs]])

            adj_mats[idxs[:, 0], 1, idxs[:, 1], idxs[:, 2]] = adj_mats[idxs[:, 0], 1, idxs[:, 2], idxs[:, 1]] = 1
            adj_mats[:, 1, :, step] = adj_mats[:, 1, step, :] = 0

            k = (last_inserted_taxa - 1)*2 - 1
            all_non_zeros = torch.nonzero(torch.triu(adj_mats[:, 1, :, :])).view(adj_mats.shape[0], k , 3)
            chosen = idxs[range(0, adj_mats.shape[0] * 2, 2)].repeat_interleave(k, dim=0).view(adj_mats.shape[0], k , 3)
            tj = torch.argwhere((all_non_zeros == chosen).prod(dim=-1))
            trajectories[:, last_inserted_taxa - 3] = tj[:, 1]

            last_inserted_taxa -= 1
        return  trajectories

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





