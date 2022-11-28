import copy
import multiprocessing as mp
import numpy as np

import concurrent.futures

import torch

from Solvers.SWA.swa_solver_torch import SwaSolverTorch


def rollout_node(node):
    return node.rollout()


class NodeTorch:

    def __init__(self, d, adj_mat, n_taxa, c=2 ** (1 / 2), step_i=3, parent=None, device=None):
        self._d = d
        self._adj_mat = adj_mat
        self._n_taxa = n_taxa
        self._step_i = step_i
        self._c = c
        self._val = None
        self._n_visits = 0

        self._parent = parent
        self._children = None

        self._device = device

    '''
    Return the best child node according to the UCT rule
    '''

    def get_mat(self):
        return self._adj_mat

    def best_child(self):
        scores = [child.value() + self._c * np.sqrt(np.log(self._n_visits) / child.n_visits())
                  for child in self._children]
        best_child = np.argmax(scores)
        return self._children[best_child]

    def value(self):
        return self._val

    def set_value(self, val):
        self._val = val

    def n_visits(self):
        return self._n_visits

    def add_visit(self):
        self._n_visits += 1

    '''
    Expand the current node by rolling out every child node and back-propagate information to parent nodes
    '''

    def expand(self):
        self._init_children()
        best_val_adj = self.rollout()

        self._val = -best_val_adj[0]
        self.add_visit()
        self._backprop()

        return best_val_adj

    '''
    Rollout this node using the given rollout policy and update node value
    '''

    def rollout(self):
        adj_mats = torch.cat([child.get_mat() for child in self._children])
        obj_vals, sol_adj_mat = self._rollout_policy(self._step_i + 1, adj_mats)
        for i, child in enumerate(self._children):
            child.set_value(-obj_vals[i])
            child.add_visit()
        idx = torch.argmin(obj_vals)
        return obj_vals[idx], sol_adj_mat[idx]

    '''
    From the given adj_mat extrapolate possible actions
    '''

    def _init_children(self):

        idxs = torch.nonzero(torch.triu(self._adj_mat), as_tuple=True)
        idxs = (torch.tensor(range(idxs[0].shape[0])).to(self._device), idxs[1], idxs[2])
        new_mats = self.add_nodes(copy.deepcopy(self._adj_mat.repeat((idxs[0].shape[0], 1, 1))),
                                  idxs, self._step_i, self._n_taxa)
        self._children = [NodeTorch(self._d, mat.unsqueeze(0), self._n_taxa, self._c, self._step_i + 1, self)
                          for mat in new_mats]

    '''
    Checks if the current node corresponds to a terminal node 
    '''

    def is_terminal(self):
        return self._step_i == self._n_taxa

    '''
    Back-propagate rollout information to parent nodes
    '''

    def _backprop(self):
        parent = self.parent()
        while parent is not None:
            if parent.value() < self.value():
                parent._val = self._val
            parent.add_visit()
            parent = parent.parent()

    '''
    Return the parent node
    '''

    def parent(self):
        return self._parent

    def is_expanded(self):
        return self._children is not None

    def _rollout_policy(self, start, adj_mats: torch.tensor = None):
        batch_size = adj_mats.shape[0]
        obj_vals = None
        for step in range(start, self._n_taxa):
            idxs_list = torch.nonzero(torch.triu(adj_mats), as_tuple=True)
            idxs_list = (torch.tensor(range(idxs_list[0].shape[0])).to(self._device), idxs_list[1], idxs_list[2])
            minor_idxs = torch.tensor([j for j in range(step + 1)]
                                      + [j for j in range(self._n_taxa, self._n_taxa + step - 1)]).to(self._device)

            repetitions = 3 + (step - 3) * 2

            adj_mats = adj_mats.repeat_interleave(repetitions, dim=0)

            sol = self.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=self._n_taxa)
            obj_vals = self.compute_obj_val_batch(adj_mats[:, minor_idxs, :][:, :, minor_idxs],
                                                  self._d[:step + +1, :step + 1].repeat(idxs_list[0].shape[0], 1, 1),
                                                  step + 1)
            obj_vals = torch.min(obj_vals.reshape(-1, repetitions), dim=-1)
            adj_mats = sol.unsqueeze(0).view(batch_size, repetitions,
                                             adj_mats.shape[1], adj_mats.shape[2])[range(batch_size), obj_vals.indices, :, :]

        return obj_vals.values, adj_mats

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

    @staticmethod
    def compute_obj_val_batch(adj_mat, d, n_taxa):
        Tau = torch.full_like(adj_mat, n_taxa)
        Tau[adj_mat > 0] = 1
        diag = torch.eye(adj_mat.shape[1]).repeat(adj_mat.shape[0], 1, 1).bool()
        Tau[diag] = 0  # diagonal elements should be zero
        for i in range(adj_mat.shape[1]):
            # The second term has the same shape as Tau due to broadcasting
            Tau = torch.minimum(Tau, Tau[:, i, :].unsqueeze(1).repeat(1, adj_mat.shape[1], 1)
                                + Tau[:, :, i].unsqueeze(2).repeat(1, 1, adj_mat.shape[1]))
        return (d * 2 ** (-Tau[:, :n_taxa, :n_taxa])).reshape(adj_mat.shape[0], -1).sum(dim=-1)
