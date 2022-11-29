import copy
import multiprocessing as mp
import numpy as np

import concurrent.futures

import torch

from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.UCTSolver.utc_utils import nni_landscape


def rollout_node(node):
    return node.rollout()


class NodeTorch:
    def __init__(self, adj_mat, step_i=3, d=None, n_taxa=None, c=None,  parent=None, rollout_=None,
                 compute_scores=None, device=None):
        self._adj_mat = adj_mat
        self._step_i = step_i
        self._n_taxa = n_taxa if n_taxa is not None else parent._n_taxa
        self._d = d if d is not None else parent._d

        self._c = c if c is not None else parent._c
        self._val = None
        self._n_visits = 0

        self._parent = parent
        self._children = None

        self._device = device
        self._average_obj_val = 0

        self._rollout_policy = rollout_ if rollout_ is not None else parent._rollout_policy
        self.compute_scores = compute_scores if compute_scores is not None else parent.compute_scores

    '''
    Return the best child node according to the UCT rule
    '''

    def get_mat(self):
        return self._adj_mat

    def best_child(self):
        scores = self.compute_scores(self)
        best_child = np.argmax(torch.tensor(scores).numpy())
        return self._children[best_child]

    def value(self):
        return self._val

    def set_value(self, val):
        self.update_average(val)
        self._val = val

    def n_visits(self):
        return self._n_visits

    def add_visit(self):
        self._n_visits += 1

    def average(self):
        return self._average_obj_val

    def update_average(self, best_val):
        self._average_obj_val = self._average_obj_val * (self._n_visits - 1) / self._n_visits \
                                + best_val / self._n_visits

    '''
    Expand the current node by rolling out every child node and back-propagate information to parent nodes
    '''

    def expand(self, iteration, nni=0):
        self._init_children()
        best_val_adj = self.rollout(iteration, nni)

        #perform nni local search
        if nni == 1 or nni == 3:
            for _ in range(5):
                expl_trees = nni_landscape(best_val_adj[1], self._n_taxa, len(best_val_adj[1]))
                obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, self._d, self._n_taxa)
                min_val = torch.min(obj_vals)
                if min_val < best_val_adj[0]:
                    best_val_adj = min_val, expl_trees[torch.argmin(obj_vals)]

        self.add_visit()
        self.set_value(best_val_adj[0])

        self._backprop()

        return best_val_adj

    '''
    Rollout this node using the given rollout policy and update node value
    '''

    def rollout(self, iteration, nni):
        adj_mats = torch.cat([child.get_mat() for child in self._children])
        obj_vals, sol_adj_mat = self._rollout_policy(self, self._step_i + 1, adj_mats, iteration)
        for i, child in enumerate(self._children):
            child.add_visit()
            if nni == 2 or nni == 3:
                for _ in range(5):
                    expl_trees = nni_landscape(sol_adj_mat[i], self._n_taxa, len(sol_adj_mat[i]))
                    obj_vals = NodeTorch.compute_obj_val_batch(expl_trees, self._d, self._n_taxa)
                    min_val = torch.min(obj_vals)
                    if min_val < obj_vals[i]:
                        obj_vals[i], sol_adj_mat[i] = min_val, expl_trees[torch.argmin(obj_vals)]
            child.set_value(obj_vals[i])

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
        self._children = [NodeTorch(mat.unsqueeze(0), self._step_i + 1, parent=self)
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
            parent.update_average(self._val)
            parent = parent.parent()

    '''
    Return the parent node
    '''

    def parent(self):
        return self._parent

    def is_expanded(self):
        return self._children is not None

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
