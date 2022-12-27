import copy
import multiprocessing as mp
import numpy as np

import concurrent.futures

import torch

from Solvers.SWA.swa_solver_torch import SwaSolverTorch
from Solvers.solver import Solver


def rollout_node(node):
    return node.rollout()


class NodeTorchBackTrack:
    def __init__(self, step_i=3, node_id=None, d=None, n_taxa=None, c=None,  parent=None, rollout_=None,
                 compute_scores=None, powers= None, device=None):
        self.id = node_id
        self.step_i = step_i
        self.n_taxa = n_taxa if n_taxa is not None else parent.n_taxa
        self.m = n_taxa*2 - 2 if n_taxa is not None else parent.m
        self.d = d if d is not None else parent.d

        self.c = c if c is not None else parent.c
        self.val = None
        self.n_visits = 0

        self.parent = parent
        self.children = {}

        self.powers = powers if powers is not None else parent.powers

        self.device = device if device is not None else parent.device
        self.average_obj_val = 0

        self.rollout_policy = rollout_ if rollout_ is not None else parent.rollout_policy
        self.compute_scores = compute_scores if compute_scores is not None else parent.compute_scores

    '''
    Return the best child node according to the UCT rule
    '''

    def best_child(self):
        scores = self.compute_scores(self)
        best_child = list(self.children.keys())[np.argmax(scores)]
        return self.children[best_child]

    def set_value(self, val):
        self.update_average(val)
        if self.val is None or val < self.val:
            self.val = val

    def add_visit(self):
        self.n_visits += 1

    def update_average(self, best_val):
        self.average_obj_val = self.average_obj_val * (self.n_visits - 1) / self.n_visits \
                                + best_val / self.n_visits

    '''
    Expand the current node by rolling out every child node and back-propagate information to parent nodes
    '''

    def expand(self, adj_mat):
        new_mats = self.init_children(adj_mat)
        obj_vals, sol_adj_mat = self.rollout(new_mats)
        idx = torch.argmin(obj_vals)
        self.update_and_backprop(obj_vals[idx])
        return obj_vals[idx], sol_adj_mat[idx], obj_vals, sol_adj_mat

    def update_and_backprop(self, best_val):
        self.add_visit()
        self.set_value(best_val)
        self.backprop()

    '''
    Rollout this node using the given rollout policy and update node value
    '''

    def rollout(self, adj_mats):
        obj_vals, sol_adj_mat = self.rollout_policy(self.step_i + 1, self.d, adj_mats, self.n_taxa, self.powers,
                                                     self.device)
        for i, child in enumerate(self.children):
            self.children[child].add_visit()
            self.children[child].set_value(obj_vals[i])
        return obj_vals, sol_adj_mat

    '''
    From the given adj_mat extrapolate possible actions
    '''

    def init_children(self, adj_mat):
        idxs = torch.nonzero(torch.triu(adj_mat), as_tuple=True)
        idxs = (torch.tensor(range(idxs[0].shape[0])).to(self.device), idxs[1], idxs[2])
        new_mats = Solver.add_nodes(copy.deepcopy(adj_mat.repeat((idxs[0].shape[0], 1, 1))),
                                  idxs, self.step_i, self.n_taxa)
        for i in range(new_mats.shape[0]):
            if i not in self.children.keys():
                self.children[i] = NodeTorchBackTrack(self.step_i + 1, node_id=i, parent=self)


        return new_mats

    '''
    Checks if the current node corresponds to a terminal node 
    '''

    def is_terminal(self):
        return self.step_i == self.n_taxa - 1

    '''
    Back-propagate rollout information to parent nodes
    '''

    def backprop(self):
        parent = self.parent
        while parent is not None:
            if parent.val < self.val:
                parent._val = self.val
            parent.add_visit()
            parent.update_average(self.val)
            parent = parent.parent

    '''
    Return the parent node
    '''


    def is_expanded(self):
        return len(self.children) >=3


    def fill(self, tj, obj_val):
        self.add_visit()
        self.set_value(obj_val)
        if not self.is_terminal():
            idx = tj[self.step_i - 3].item()
            if idx in self.children.keys():
                self.children[idx].fill(tj, obj_val)
            else:
                self.children[idx] = NodeTorchBackTrack(self.step_i + 1, node_id=idx, parent=self)
                self.children[idx].fill(tj, obj_val)







