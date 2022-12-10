import copy
import multiprocessing as mp
import numpy as np

import concurrent.futures

import torch

from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.solver import Solver


def rollout_node(node):
    return node.rollout()


class NodeTorch:
    def __init__(self, adj_mat, step_i=3, d=None, n_taxa=None, c=None,  parent=None, rollout_=None,
                 compute_scores=None, device=None):
        self._adj_mat = adj_mat
        self._step_i = step_i
        self._n_taxa = n_taxa if n_taxa is not None else parent._n_taxa
        self._m = n_taxa*2 - 2 if n_taxa is not None else parent._m
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
        best_child = np.argmax(scores)
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

    def expand(self, iteration):
        self._init_children()
        # print("n children", len(self._children))
        obj_vals, sol_adj_mat = self.rollout(iteration)
        idx = torch.argmin(obj_vals)
        self.update_and_backprop(obj_vals[idx])
        return obj_vals[idx], sol_adj_mat[idx]

    def expand_full(self, iteration):
        self._init_children()
        # print("n children", len(self._children))
        return self.rollout(iteration)
    def second_expand(self, swa_nni_policy):
        adj_mats = torch.cat([child.get_mat() for child in self._children])
        return swa_nni_policy(self, self._step_i + 1, adj_mats)

    def update_and_backprop(self, best_val):
        self.add_visit()
        self.set_value(best_val)
        self._backprop()

    def nni_expand(self, iteration):
        pass
    '''
    Rollout this node using the given rollout policy and update node value
    '''

    def rollout(self, iteration):
        adj_mats = torch.cat([child.get_mat() for child in self._children])
        obj_vals, sol_adj_mat = self._rollout_policy(self._step_i + 1, self._d, adj_mats, self._n_taxa, iteration)
        for i, child in enumerate(self._children):
            child.add_visit()
            child.set_value(obj_vals[i])
        return obj_vals, sol_adj_mat

    '''
    From the given adj_mat extrapolate possible actions
    '''

    def _init_children(self):
        idxs = torch.nonzero(torch.triu(self._adj_mat), as_tuple=True)
        idxs = (torch.tensor(range(idxs[0].shape[0])).to(self._device), idxs[1], idxs[2])
        new_mats = Solver.add_nodes(copy.deepcopy(self._adj_mat.repeat((idxs[0].shape[0], 1, 1))),
                                  idxs, self._step_i, self._n_taxa)
        self._children = [NodeTorch(mat.unsqueeze(0), self._step_i + 1, parent=self)
                          for mat in new_mats]

    '''
    Checks if the current node corresponds to a terminal node 
    '''

    def is_terminal(self):
        return self._step_i == self._n_taxa - 1

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



