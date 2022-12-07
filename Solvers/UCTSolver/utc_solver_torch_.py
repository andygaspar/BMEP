import time

import numpy as np
import torch

from Solvers.FastME.fast_me import FastMeSolver
from Solvers.UCTSolver.node_torch import NodeTorch
from Solvers.UCTSolver.utils.utc_utils import run_nni_search
from Solvers.UCTSolver.utils.utils_rollout import swa_policy
from Solvers.UCTSolver.utils.utils_scores import max_score_normalised
from Solvers.solver import Solver


class UtcSolverTorchSingleBackTrack(Solver):

    def __init__(self, d: np.array, rollout_, compute_scores, c_initial=2 ** (1 / 2), nni_tol=0.02):
        super(UtcSolverTorchSingleBackTrack, self).__init__(d)
        self.numpy_d = self.d

        self.rollout_ = rollout_
        self.compute_scores = compute_scores
        self.adj_mat_sparse = None
        self.root = None
        self.init_c = c_initial
        self.n_nodes = 0

        self.fast_me = FastMeSolver(d, bme=True, nni=True, digits=17, post_processing=True,
                                    triangular_inequality=False, logs=False)
        self.nni_tol = 1 + nni_tol

        self.backtracking_time = 0

    def solve(self, n_iterations=100):
        # with torch.no_grad():
        self.d = torch.tensor(self.d, requires_grad=False).to(self.device)
        adj_mat = self.initial_adj_mat(device=self.device, n_problems=1)
        self.root = NodeTorch(adj_mat, step_i=3, d=self.d, n_taxa=self.n_taxa, c=self.init_c, parent=None,
                              rollout_=self.rollout_, compute_scores=self.compute_scores, device=self.device)
        self.obj_val, self.solution = self.root.expand(0)

        for iteration in range(n_iterations):
            node = self.root
            while not node.is_terminal() and node.is_expanded():
                node = node.best_child()

            if node.is_terminal():
                break
            run_val, run_sol = node.expand(iteration)

            t = time.time()
            Tau = self.get_tau_tensor(run_sol, self.n_taxa)
            self.fast_me.update_topology(Tau)
            self.fast_me.solve()
            if self.fast_me.obj_val < run_val:
                run_val, run_sol = self.fast_me.obj_val, torch.tensor(self.fast_me.solution)
                trajectory_id = self.tree_climb(run_sol)
                node_climbed = self.get_lower_node_in_trajectory(trajectory_id)
                node_climbed.update_and_backprop(run_val)

            self.backtracking_time += time.time() - t
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

    def initializer(self, init_steps):
        adj_mats = self.initial_adj_mat(device=self.device, n_problems=1)
        self.root = NodeTorch(adj_mats, step_i=3, d=self.d, n_taxa=self.n_taxa, c=self.init_c, parent=None,
                              rollout_=self.rollout_, compute_scores=self.compute_scores, device=self.device)
        self.d = torch.tensor(self.d, requires_grad=False).to(self.device)
        nodes_level = [self.root]

        for step in range(3, init_steps):
            idxs_list = torch.nonzero(torch.triu(adj_mats), as_tuple=True)
            idxs_list = (torch.tensor(range(idxs_list[0].shape[0])).to(self.device), idxs_list[1], idxs_list[2])
            repetitions = 3 + (step - 3) * 2
            adj_mats = adj_mats.repeat_interleave(repetitions, dim=0)
            adj_mats = self.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=self.n_taxa)

            self.root._children = [NodeTorch(mat.unsqueeze(0), step + 1, parent=self)
                          for mat in adj_mats]

        for step in range(init_steps, self.n_taxa):
            idxs_list = torch.nonzero(torch.triu(adj_mats), as_tuple=True)
            idxs_list = (torch.tensor(range(idxs_list[0].shape[0])).to(self.device), idxs_list[1], idxs_list[2])
            minor_idxs = torch.tensor([j for j in range(step + 1)]
                                      + [j for j in range(self.n_taxa, self.n_taxa + step - 1)]).to(self.device)

            repetitions = 3 + (step - 3) * 2

            adj_mats = adj_mats.repeat_interleave(repetitions, dim=0)

            sol = self.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=self.n_taxa)
            obj_vals = self.compute_obj_val_batch(adj_mats[:, minor_idxs, :][:, :, minor_idxs],
                                               self.d[:step + 1, :step + 1].repeat(idxs_list[0].shape[0], 1, 1),
                                               step + 1)
            obj_vals = torch.min(obj_vals.reshape(-1, repetitions), dim=-1)
            adj_mats = sol.unsqueeze(0).view(15, repetitions, adj_mats.shape[1],
                                             adj_mats.shape[2])[range(15), obj_vals.indices, :, :]

        return obj_vals.values, adj_mats
# u = UtcSolverTorchBackTrack(np.random.uniform(size=(9, 9)), swa_policy, max_score_normalised)
# vals, mats = u.initializer()
# print("done")


