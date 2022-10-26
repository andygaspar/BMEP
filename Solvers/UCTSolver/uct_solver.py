import numpy as np


class Node:

    def __init__(self, state, n_taxa, step_i, c, parent=None):
        self._state = state
        self._n_taxa = n_taxa
        self._step_i = step_i
        self._c = c

        self._val = None
        self._n_visits = 0

        self._parent = parent
        self._children = []

    '''
    Return the best child node according to the UCT rule
    '''
    def best_child(self):
        best_child = np.argmax(
            [child.value() + self._c * np.sqrt(np.log(self._n_visits) / child.n_visits()) for child in self._children])
        return self._children[best_child]

    def value(self):
        return self._val

    def n_visits(self):
        return self._n_visits

    def add_visit(self):
        self._n_visits += 1

    '''
    Expand the current node by rolling out every child node and back-propagate information to parent nodes
    '''

    def expand(self, rollout_policy):
        self._init_children()
        map(lambda x: x.rollout(rollout_policy), self._children)
        self._val = max([c.value() for c in self._children])

        self._backprop()

    '''
    Rollout this node using the given rollout policy and update node value
    '''
    def rollout(self, rollout_policy):
        current_state = self._state
        for i in range(self._step_i, self._n_taxa):
            action = rollout_policy(current_state)
            ### UPDATE STATE ###
            state = ...
            ####################
        self._val = compute_obj_val()
        self.add_visit()


    '''
    From the given state extrapolate possible actions
    '''

    def _init_children(self):
        self._children = list(zip(*np.nonzero(np.triu(self._state))))

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
        while parent is not None and parent.value() < self.value():
            parent._val = self._val
            parent = parent.parent()

    '''
    Return the parent node
    '''

    def parent(self):
        return self._parent
