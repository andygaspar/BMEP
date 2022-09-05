import networkx as nx
import numpy as np
import torch


class Solver:

    def __init__(self, d=None):
        self.d = d
        self.m = d.shape[0] if d is not None else None
        self.n = (self.m + 2) // 2 if d is not None else None

        self.solution = None

    def initial_mat(self, device=None):
        adj_mat = np.zeros_like(self.d) if device is None else torch.zeros_like(self.d).to(device)
        adj_mat[0, self.n] = adj_mat[self.n, 0] = 1
        adj_mat[1, self.n] = adj_mat[self.n, 1] = 1
        adj_mat[2, self.n] = adj_mat[self.n, 2] = 1
        return adj_mat

    @staticmethod
    def add_node(adj_mat, idxs, new_node_idx, n):
        adj_mat[idxs[0], idxs[1]] = adj_mat[idxs[1], idxs[0]] = 0  # detach selected
        adj_mat[idxs[0], n + new_node_idx - 2] = adj_mat[n + new_node_idx - 2, idxs[0]] = 1  # reattach selected to new
        adj_mat[idxs[1], n + new_node_idx - 2] = adj_mat[n + new_node_idx - 2, idxs[1]] = 1  # reattach selected to new
        adj_mat[new_node_idx, n + new_node_idx - 2] = adj_mat[n + new_node_idx - 2, new_node_idx] = 1  # attach new

        return adj_mat

    @staticmethod
    def compute_obj_val_from_adj_mat(adj_mat, d, n):
        g = nx.from_numpy_matrix(adj_mat)
        Tau = nx.floyd_warshall_numpy(g)[:n, :n]
        return np.sum([d[i, j] / 2 ** (Tau[i, j]) for i in range(n) for j in range(n)])