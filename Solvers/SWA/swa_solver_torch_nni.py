import copy

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from Data_.Datasets.bmep_dataset import BMEP_Dataset
from Solvers.UCTSolver.utils.utc_utils import run_nni_search
from Solvers.solver import Solver


class SwaSolverTorchNni(Solver):

    def __init__(self, d, sorted_d=False):
        super(SwaSolverTorchNni, self).__init__(d, sorted_d)
        self.d = torch.tensor(self.d).to(self.device)
        self.n_internals = self.m - self.n_taxa

    def solve(self, start=3, adj_mat: torch.tensor = None, every=10, after=20, n_iter=5, n_final_iter=20):
        adj_mat = self.initial_adj_mat(self.device, n_problems=1) if adj_mat is None else adj_mat
        obj_vals, best_idx = None, None
        for step in range(start, self.n_taxa):

            idxs_list = torch.nonzero(torch.triu(adj_mat), as_tuple=True)
            idxs_list = (torch.tensor(range(idxs_list[0].shape[0])).to(self.device), idxs_list[1], idxs_list[2])
            minor_idxs = torch.tensor([j for j in range(step + 1)]
                                      + [j for j in range(self.n_taxa, self.n_taxa + step - 1)]).to(self.device)

            adj_mat = adj_mat.repeat((idxs_list[0].shape[0], 1, 1))

            sol = self.add_nodes(adj_mat, idxs_list, new_node_idx=step, n=self.n_taxa)
            obj_vals = self.compute_obj_val_batch(adj_mat[:, minor_idxs, :][:, :, minor_idxs],
                                                  self.d[:step + +1, :step + 1].repeat((idxs_list[0].shape[0], 1, 1)),
                                                  powers=self.powers, n_taxa=step + 1, device=self.device)

            best_idx = torch.argmin(obj_vals)
            adj_mat = sol[best_idx].unsqueeze(0)
            # if step > after and (step - 3) % every == 0:
            #
            #     improved, best_val, best_solution = \
            #         run_nni_search(adj_mat.squeeze(0), obj_vals[best_idx], self.d, self.n_taxa, self.m, self.powers, self.device)
            #
            #     if improved:
            #         adj_mat = self.adjust_matrix(best_solution, step)

        improved, best_val, best_solution = \
            run_nni_search(adj_mat.squeeze(0), obj_vals[best_idx], self.d, self.n_taxa, self.m, powers=self.powers, device=self.device)

        if improved:
            adj_mat = self.adjust_matrix(best_solution, self.n_taxa - 1)
        self.solution = adj_mat.squeeze(0)
        self.obj_val = torch.min(obj_vals).item() if not improved else best_val.item()

    @staticmethod
    def add_nodes(adj_mat, idxs: torch.tensor, new_node_idx, n):
        adj_mat[idxs] = adj_mat[idxs[0], idxs[2], idxs[1]] = 0  # detach selected
        adj_mat[idxs[0], idxs[1], n + new_node_idx - 2] = adj_mat[
            idxs[0], n + new_node_idx - 2, idxs[1]] = 1  # reattach selected to new
        adj_mat[idxs[0], idxs[2], n + new_node_idx - 2] = adj_mat[
            idxs[0], n + new_node_idx - 2, idxs[2]] = 1  # reattach selected to new
        adj_mat[idxs[0], new_node_idx, n + new_node_idx - 2] = adj_mat[
            idxs[0], n + new_node_idx - 2, new_node_idx] = 1  # attach new

        return adj_mat

    # def compute_obj_val_batch(self, adj_mat, d, n_taxa):
    #     A = torch.full_like(adj_mat, n_taxa)
    #     A[adj_mat > 0] = 1
    #     diag = torch.eye(adj_mat.shape[1]).repeat(adj_mat.shape[0], 1, 1).bool()
    #     A[diag] = 0  # diagonal elements should be zero
    #     for i in range(adj_mat.shape[1]):
    #         # The second term has the same shape as A due to broadcasting
    #         A = torch.minimum(A, A[:, i, :].unsqueeze(1).repeat(1, adj_mat.shape[1], 1)
    #                           + A[:, :, i].unsqueeze(2).repeat(1, 1, adj_mat.shape[1]))
    #     return (d * 2**(-A[:, :n_taxa, :n_taxa])).reshape(adj_mat.shape[0], -1).sum(dim=-1)

    def adjust_matrix(self, adj_mat, last_inserted_taxa):
        adj_mat = adj_mat.unsqueeze(0).repeat(2, 1, 1)

        # reorder matrix according to Pardi
        for step in range(last_inserted_taxa + self.n_internals, self.n_taxa, -1):

            if adj_mat[1][step, last_inserted_taxa] == 0:
                idx = torch.nonzero(adj_mat[1][last_inserted_taxa]).item()
                adj_mat = self.permute(adj_mat, step, idx)

            adj_mat[1][:, last_inserted_taxa] = adj_mat[1][last_inserted_taxa, :] = 0
            idxs = torch.nonzero(adj_mat[1][step])
            adj_mat[1][idxs[0], idxs[1]] = adj_mat[1][idxs[1], idxs[0]] = 1
            adj_mat[1][:, step] = adj_mat[1][step, :] = 0

            last_inserted_taxa -= 1
        return adj_mat[0].unsqueeze(0)

    @staticmethod
    def permute(adj_mats, step, idx):
        adj_mats[:, step, :] += adj_mats[:, idx, :]
        adj_mats[:, idx, :] = adj_mats[:, step, :] - adj_mats[:, idx, :]
        adj_mats[:, step, :] -= adj_mats[:, idx, :]

        adj_mats[:, :, step] += adj_mats[:, :, idx]
        adj_mats[:, :, idx] = adj_mats[:, :, step] - adj_mats[:, :, idx]
        adj_mats[:, :, step] -= adj_mats[:, :, idx]
        return adj_mats
