import copy

import networkx as nx
import numpy as np
import torch


class Solver:

    def __init__(self, d=None, sorted_d=False):
        if sorted_d:
            self.d = d
        else:
            self.d = self.sort_d(d) if d is not None else None
        self.n_taxa = d.shape[0] if d is not None else None
        self.m = self.n_taxa * 2 - 2 if d is not None else None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.solution = None
        self.obj_val = None
        self.T = None

    @staticmethod
    def sort_d(d):
        is_tensor = type(d) == torch.Tensor
        dist_sum = np.sum(d, axis=0) if not is_tensor else torch.sum(d, dim=-1)
        if not is_tensor:
            order = np.flip(np.argsort(dist_sum))
        else:
            order = torch.flip(torch.argsort(dist_sum), dims=(0,))
        sorted_d = np.zeros_like(d) if not is_tensor else torch.zeros_like(d)
        for i in order:
            for j in order:
                sorted_d[i, j] = d[order[i], order[j]]
        return sorted_d

    def initial_adj_mat(self, device=None):
        adj_mat = np.zeros((self.m, self.m)) if device is None else torch.zeros((self.m, self.m)).to(device)
        adj_mat[0, self.n_taxa] = adj_mat[self.n_taxa, 0] = 1
        adj_mat[1, self.n_taxa] = adj_mat[self.n_taxa, 1] = 1
        adj_mat[2, self.n_taxa] = adj_mat[self.n_taxa, 2] = 1
        return adj_mat

    @staticmethod
    def get_tau(adj_mat, device=None):
        g = nx.from_numpy_matrix(adj_mat)
        tau = nx.floyd_warshall_numpy(g)
        tau[np.isinf(tau)] = 0
        return tau if device is None else torch.tensor(tau).to(torch.float).to(device)

    @staticmethod
    def add_node(adj_mat, idxs, new_node_idx, n):
        adj_mat[idxs[0], idxs[1]] = adj_mat[idxs[1], idxs[0]] = 0  # detach selected
        adj_mat[idxs[0], n + new_node_idx - 2] = adj_mat[n + new_node_idx - 2, idxs[0]] = 1  # reattach selected to new
        adj_mat[idxs[1], n + new_node_idx - 2] = adj_mat[n + new_node_idx - 2, idxs[1]] = 1  # reattach selected to new
        adj_mat[new_node_idx, n + new_node_idx - 2] = adj_mat[n + new_node_idx - 2, new_node_idx] = 1  # attach new

        return adj_mat

    def compute_obj(self):
        return np.sum([self.d[i, j] / 2 ** (self.T[i, j]) for i in range(self.n_taxa) for j in range(self.n_taxa)])

    @staticmethod
    def compute_obj_val_from_adj_mat(adj_mat, d, n):
        g = nx.from_numpy_matrix(adj_mat)
        Tau = nx.floyd_warshall_numpy(g)[:n, :n]
        return np.sum([d[i, j] / 2 ** (Tau[i, j]) for i in range(n) for j in range(n)])




