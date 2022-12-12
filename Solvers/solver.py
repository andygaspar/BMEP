import copy
import time
from abc import abstractmethod

import networkx as nx
import numpy as np
import torch

pippo = 0

class Solver:

    def __init__(self, d=None, sorted_d=False):
        if sorted_d:
            self.d = d
        else:
            self.d = self.sort_d(d) if d is not None else None
        self.n_taxa = d.shape[0] if d is not None else None
        self.m = self.n_taxa * 2 - 2 if d is not None else None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.powers = np.array([2 ** (-i) for i in range(self.n_taxa)]) if self.n_taxa is not None else None

        self.solution = None
        self.obj_val = None
        self.T = None

        self.time = None

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

    def initial_adj_mat(self, device=None, n_problems=0):
        if n_problems == 0:
            adj_mat = np.zeros((self.m, self.m)) if device is None else torch.zeros((self.m, self.m)).to(device)
            adj_mat[0, self.n_taxa] = adj_mat[self.n_taxa, 0] = 1
            adj_mat[1, self.n_taxa] = adj_mat[self.n_taxa, 1] = 1
            adj_mat[2, self.n_taxa] = adj_mat[self.n_taxa, 2] = 1
        else:
            adj_mat = torch.zeros((n_problems, self.m, self.m)).to(self.device)
            adj_mat[:, 0, self.n_taxa] = adj_mat[:, self.n_taxa, 0] = 1
            adj_mat[:, 1, self.n_taxa] = adj_mat[:, self.n_taxa, 1] = 1
            adj_mat[:, 2, self.n_taxa] = adj_mat[:, self.n_taxa, 2] = 1
            return adj_mat
        return adj_mat

    @abstractmethod
    def solve(self, *args):
        pass

    def solve_timed(self, *args):
        self.time = time.time()
        self.solve(*args)
        self.time = time.time() - self.time

    def get_tau(self, adj_mat):
        T = np.full(adj_mat.shape, self.n_taxa)
        T[adj_mat > 0] = 1
        np.fill_diagonal(T, 0)  # diagonal elements should be zero
        for i in range(adj_mat.shape[0]):
            # The second term has the same shape as T due to broadcasting
            T = np.minimum(T, T[i, :][np.newaxis, :] + T[:, i][:, np.newaxis])
        T = T[:self.n_taxa, :self.n_taxa]
        return T

    @staticmethod
    def get_tau_tensor(adj_mat, n_taxa):
        Tau = torch.full_like(adj_mat, n_taxa)
        Tau[adj_mat > 0] = 1
        diag = torch.eye(adj_mat.shape[1]).bool()
        Tau[diag] = 0  # diagonal elements should be zero
        for i in range(adj_mat.shape[1]):
            # The second term has the same shape as Tau due to broadcasting
            Tau = torch.minimum(Tau, Tau[ i, :].unsqueeze(0)
                                + Tau[:, i].unsqueeze(1))
        return Tau[:n_taxa, :n_taxa]

    @staticmethod
    def add_node(adj_mat, idxs, new_node_idx, n):
        adj_mat[idxs[0], idxs[1]] = adj_mat[idxs[1], idxs[0]] = 0  # detach selected
        adj_mat[idxs[0], n + new_node_idx - 2] = adj_mat[n + new_node_idx - 2, idxs[0]] = 1  # reattach selected to new
        adj_mat[idxs[1], n + new_node_idx - 2] = adj_mat[n + new_node_idx - 2, idxs[1]] = 1  # reattach selected to new
        adj_mat[new_node_idx, n + new_node_idx - 2] = adj_mat[n + new_node_idx - 2, new_node_idx] = 1  # attach new

        return adj_mat

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

    def compute_obj(self):
        return np.sum(
            [self.d[i, j] / self.powers[self.T[i, j]] for i in range(self.n_taxa) for j in range(self.n_taxa)])

    def compute_obj_val_from_adj_mat(self, adj_mat, d, n_taxa):
        T = np.full(adj_mat.shape, n_taxa)
        T[adj_mat > 0] = 1
        np.fill_diagonal(T, 0)  # diagonal elements should be zero
        for i in range(adj_mat.shape[0]):
            # The second term has the same shape as T due to broadcasting
            T = np.minimum(T, T[i, :][np.newaxis, :] + T[:, i][:, np.newaxis])
        T = T[:n_taxa, :n_taxa]
        r = range(n_taxa)
        return np.sum([d[i, j] / self.powers[T[i, j]] for i in r for j in r])

    @staticmethod
    def compute_obj_val_batch(adj_mat, d, n_taxa):
        vals = torch.tensor([2**(-i) for i in range(n_taxa)], device='cuda:0')
        t = time.time()
        Tau = torch.full_like(adj_mat, n_taxa)
        Tau[adj_mat > 0] = 1
        diag = torch.eye(adj_mat.shape[1], device='cuda:0').repeat(adj_mat.shape[0], 1, 1).bool()
        Tau[diag] = 0  # diagonal elements should be zero
        for i in range(adj_mat.shape[1]):
            # The second term has the same shape as Tau due to broadcasting
            Tau = torch.minimum(Tau, Tau[:, i, :].unsqueeze(1).repeat(1, adj_mat.shape[1], 1)
                                + Tau[:, :, i].unsqueeze(2).repeat(1, 1, adj_mat.shape[1]))
        t = time.time() - t
        tt = time.time()
        val = (d * 2 ** (-Tau[:, :n_taxa, :n_taxa])).reshape(adj_mat.shape[0], -1).sum(dim=-1)
        tt = time.time() - tt

        Tau = Tau.to(torch.long)
        ttt = time.time()
        val2 = (d * vals[Tau[:, :n_taxa, :n_taxa]]).reshape(adj_mat.shape[0], -1).sum(dim=-1)
        ttt = time.time() - ttt
        print("vvv",t, tt, ttt)
        return val, t, tt, ttt
    @staticmethod
    def compute_obj_val_batch_2(adj_mat, d, n_taxa):
        sub_adj = adj_mat[n_taxa:, n_taxa:]
        Tau_int = torch.full_like(sub_adj, n_taxa)
        Tau_int[sub_adj > 0] = 1
        diag = torch.eye(sub_adj.shape[1], device='cuda:0').repeat(adj_mat.shape[0], 1, 1).bool()
        Tau_int[diag] = 0  # diagonal elements should be zero
        for i in range(sub_adj.shape[1]):
            # The second term has the same shape as Tau_int due to broadcasting
            Tau_int = torch.minimum(Tau_int, Tau_int[i, :].unsqueeze(0)
                                + Tau_int[:, i].unsqueeze(1))
        idxs = torch.nonzero(adj_mat[:n_taxa])[:, 1]
        idx_int_to_taxa = torch.vstack([idxs.repeat_interleave(n_taxa), idxs.repeat(n_taxa)]) - n_taxa
        Tau = (Tau_int[idx_int_to_taxa[0], idx_int_to_taxa[1]] + 2).reshape(n_taxa, n_taxa)
        return (d * 2 ** (-Tau[:, :n_taxa, :n_taxa])).reshape(adj_mat.shape[0], -1).sum(dim=-1)




