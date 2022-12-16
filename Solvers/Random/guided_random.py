import random

import numpy as np
import torch

from Solvers.Random.random_solver import RandomSolver
from Solvers.UCTSolver.utils.utc_utils import nni_landscape
from Solvers.UCTSolver.utils.utc_utils_batch import nni_landscape_batch


class GuidedRandSolver(RandomSolver):

    def __init__(self, d, iterations, sorted_d=False):
        super(GuidedRandSolver, self).__init__(d, sorted_d)
        self.d = torch.tensor(self.d, device=self.device)
        self.iterations = iterations
        self.n_nodes = 0
        self.max_depth = 0
        self.n_trajectories = 0

    def solve(self, start=3, adj_mats = None):
        adj_mats, best_val = None, 1000
        trajectories, tj = [], np.array([])
        for _ in range(self.iterations):
            adj_mats = self.initial_adj_mat(self.device, n_problems=1)
            batch_size = adj_mats.shape[0]
            current_tj = ()
            for step in range(start, self.n_taxa):
                choices = 3 + (step - 3) * 2
                idxs_list = torch.nonzero(torch.triu(adj_mats)).reshape(batch_size, -1, 3)
                rand_idxs = random.choices(range(choices), k=batch_size)[0]
                # rand_idxs, tj = self.get_choices(choices, batch_size, tj)
                current_tj += (rand_idxs,)
                idxs_list = idxs_list[[range(batch_size)], rand_idxs, :]
                idxs_list = (idxs_list[:, :, 0], idxs_list[:, :, 1], idxs_list[:, :, 2])
                adj_mats = self.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=self.n_taxa)

            trajectories.append(current_tj)

            obj_vals = self.compute_obj_val_batch(adj_mats, self.d, self.powers, self.n_taxa, self.device)
            run_val = torch.min(obj_vals)
            print("init val", run_val)
            if run_val < best_val:
                best_val = run_val


            sol = adj_mats
            improved = True
            while improved:
                expl_trees = nni_landscape_batch(sol, self.n_taxa, self.m)
                expl_trees_batch = expl_trees.view(batch_size * expl_trees.shape[1], self.m, self.m)
                obj_vals_new = self.compute_obj_val_batch(expl_trees_batch, self.d, self.powers, self.n_taxa, self.device)

                new_obj_val = torch.min(obj_vals_new.view(batch_size, -1), dim=-1)
                nni_run_val = torch.min(new_obj_val[0])
                if nni_run_val < run_val:
                    sol = expl_trees[range(batch_size), new_obj_val[1]]
                    for tree in expl_trees_batch:
                        trajectories.append(self.tree_climb(tree))
                    run_val = nni_run_val
                    if run_val < best_val:
                        best_val = run_val
                        print(best_val)
                else:
                    improved = False


            tj = np.array(trajectories)

        self.n_trajectories = len(trajectories)



        self.obj_val = best_val.item()

        self.count_nodes(tj)

    def recursion_counter(self, i, tj, depth):
        choices = 3 + i * 2
        length = np.unique(tj[:, 0]).shape[0]
        if length < choices:
            return 0, depth
        counter, deepest = choices, depth
        depth += 1
        for ii in range(choices):
            chosen = np.where(tj[:, 0] == ii)
            tjj = tj[chosen]
            tjj = tjj[:, 1:]
            count, new_depth = self.recursion_counter(i + 1, tjj, depth)
            counter += count
            if new_depth > deepest:
                deepest = new_depth

        return counter, deepest

    def count_nodes(self, tj):
        self.n_nodes, self.max_depth = self.recursion_counter(0, tj, 0)

    def get_choices(self, choices, batch_size, tj):
        if tj.shape[0] == 0:
            return random.choices(range(choices), k=batch_size)[0], tj
        else:
            vals, counts = np.unique(tj[: ,0], return_counts=True)
            missing_vals = [val for val in range(choices) if val not in vals]
            if missing_vals:
                vals = np.append(vals, missing_vals)
                counts = np.append(counts, [0 for _ in missing_vals])
            counts = counts.astype('float64')
            p = 1 - counts / counts.sum()
            p /= p.sum()
            idxs = np.random.choice(vals, size=1, replace=True, p=p)[0]
            chosen = np.where(tj[:, 0] == idxs)
            tj = tj[chosen]
            tj = tj[:, 1:]
            return idxs, tj

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
