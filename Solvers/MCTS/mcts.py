import copy

import networkx as nx
import numpy as np
from mcts import mcts
from Solvers.solver import Solver


class State:

    def __init__(self, d, adj_mat, n_taxa, step=3):
        self.adj_mat = adj_mat
        self.n_taxa = n_taxa
        self.step = step
        self.d = d

    def getPossibleActions(self):
        return list(zip(*np.nonzero(np.triu(self.adj_mat))))

    def isTerminal(self):
        return np.any(self.adj_mat[:, self.n_taxa - 1])

    def getReward(self):
        g = nx.from_numpy_matrix(self.adj_mat)
        Tau = nx.floyd_warshall_numpy(g)[:self.n_taxa, :self.n_taxa]
        return np.sum([self.d[i, j] / 2 ** (Tau[i, j]) for i in range(self.n_taxa) for j in range(self.n_taxa)])

    def takeAction(self, action):
        new_state = copy.deepcopy(self)
        new_state.adj_mat[action[0], action[1]] = new_state.adj_mat[action[1], action[0]] = 0  # detach selected
        new_state.adj_mat[action[0], self.n_taxa + new_state.step - 2] = \
            new_state.adj_mat[self.n_taxa + new_state.step - 2, action[0]] = 1  # reattach selected to new
        new_state.adj_mat[action[1], self.n_taxa + new_state.step - 2] = \
            new_state.adj_mat[self.n_taxa + new_state.step - 2, action[1]] = 1  # reattach selected to new
        new_state.adj_mat[new_state.step, self.n_taxa + new_state.step - 2] = \
            new_state.adj_mat[self.n_taxa + new_state.step - 2, new_state.step] = 1  # attach new
        new_state.step += 1
        return new_state


class MctsSolver(Solver):

    def __init__(self, d):
        super(MctsSolver, self).__init__(d)

    def solve(self, time_limit=1, rolloutPolicy=None):
        state = State(self.d, self.initial_mat(), self.n)
        searcher = mcts(timeLimit=time_limit) if rolloutPolicy is None \
            else mcts(timeLimit=time_limit, rolloutPolicy=rolloutPolicy)
        for i in range(3, self.n):
            action = searcher.search(initialState=state)
            state = state.takeAction(action)
        self.solution = state.adj_mat
        self.obj_val = state.getReward()