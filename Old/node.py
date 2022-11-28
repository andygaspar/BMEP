import copy
import multiprocessing as mp
import numpy as np

import concurrent.futures


def rollout_node(node):
    return node.rollout()


class Node:

    def __init__(self, adj_mat, n_taxa, add_node, rollout_policy, c=2 ** (1 / 2), step_i=3, parent=None, to_tensor=False):
        self._adj_mat = adj_mat
        self._n_taxa = n_taxa
        self._step_i = step_i
        self._c = c
        self.update_adj_mat = add_node
        self._val = None
        self._n_visits = 0

        self._parent = parent
        self._children = None

        self._rollout_policy = rollout_policy
        self._to_tensor = to_tensor

        self.expand = self.expand_seq if self._n_taxa < 10 or self._to_tensor else self.expand_parallel


    '''
    Return the best child node according to the UCT rule
    '''

    def best_child(self):
        best_child = np.argmax(
            [child.value() + self._c * np.sqrt(np.log(self._n_visits) / child.n_visits()) for child in self._children])
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

    def expand_parallel(self):
        self._init_children()

        num_procs = mp.cpu_count()
        pool = mp.Pool(num_procs)
        child_results = pool.map(rollout_node, self._children)
        pool.close()
        pool.join()
        for child, result in zip(self._children, child_results):
            child.set_value(-result[0])
            child.add_visit()

        self._val = max([c.value() for c in self._children])
        self.add_visit()
        self._backprop()

        best_idx = np.argmin([child[0] for child in child_results])

        return child_results[best_idx]

    def expand_seq(self):
        self._init_children()
        child_results = list(map(lambda x: x.rollout(), self._children))

        self._val = max([c.value() for c in self._children])
        self.add_visit()
        self._backprop()

        best_idx = np.argmin([child[0] for child in child_results])

        return child_results[best_idx]

    '''
    Rollout this node using the given rollout policy and update node value
    '''

    def rollout(self):
        obj_val, sol_adj_mat = self._rollout_policy(self._adj_mat, self._step_i)
        self._val = -obj_val
        self.add_visit()
        return obj_val, sol_adj_mat

    '''
    From the given adj_mat extrapolate possible actions
    '''

    def _init_children(self):

        actions = list(zip(*np.nonzero(np.triu(self._adj_mat))))
        attributes = (self._n_taxa, self.update_adj_mat, self._rollout_policy,
                      self._c, self._step_i + 1, self, self._to_tensor)
        self._children = [Node(self.update_adj_mat(copy.deepcopy(self._adj_mat), act, self._step_i, self._n_taxa),
                               *attributes) for act in actions]

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
