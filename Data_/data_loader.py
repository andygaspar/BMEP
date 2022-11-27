import copy
import random
import warnings
from os import walk
from typing import Union

import networkx as nx
import numpy as np
from gurobipy import Model, GRB, quicksum, Env
from matplotlib import pyplot as plt

from Solvers.Random.random_solver import RandomSolver

warnings.simplefilter("ignore")


class DataSet:

    def __init__(self, d, name):
        self.d = d
        self.name = name
        self.size = self.d.shape[0]

    def __repr__(self):
        return self.name

    def get_full_dataset(self):
        return self.d

    def get_minor(self, *args):
        if len(args) > 1:
            from_, to_ = args
            return self.d[from_: to_, from_: to_]
        else:
            to_ = args[0]
            return self.d[: to_, : to_]

    def get_random_mat(self, dim, from_=None, to_=None):
        from_ = from_ if from_ is not None else 0
        to_ = to_ if to_ is not None else self.size
        idx = random.sample(range(from_, to_), k=dim)
        return self.d[idx, :][:, idx]


class DistanceData:

    def __init__(self):
        path = 'Data_/csv_'
        filenames = sorted(next(walk(path), (None, None, []))[2])

        self.data_sets = {}
        for file in filenames:
            if file[-4:] == '.txt':
                self.data_sets[file[:-4]] = DataSet(np.genfromtxt('Data_/csv_/' + file, delimiter=','), file[:-4])

    def print_dataset_names(self):
        for key in self.data_sets.keys():
            print(key)

    def get_dataset_names(self):
        return list(self.data_sets.keys())

    def get_dataset(self, data_set: Union[str, int]):
        if type(data_set) == int:
            return list(self.data_sets.values())[data_set]
        else:
            return self.data_sets[data_set]

    @staticmethod
    def generate_random(dim, a=0, b=1):
        d = np.random.uniform(a, b, (dim, dim))
        np.fill_diagonal(d, 0)
        return np.triu(d) + np.triu(d).T

    @staticmethod
    def generate_random_square(dim, a=0, b=1):
        d = np.zeros((dim, dim))
        points = np.random.uniform(a, b, (dim, 2))
        for i in range(dim):
            d[i, :] = np.sqrt(np.sum((points[i] - points) ** 2, axis=1))
        return d

    @staticmethod
    def generate_double_stochastic(dim):
        m = Model()
        d = m.addMVar((dim, dim), vtype=GRB.CONTINUOUS)
        c = np.random.uniform(size=(dim, dim))
        m.addConstr(d @ np.ones(dim) == np.ones(dim))
        m.addConstr(d.T @ np.ones(dim) == np.ones(dim))
        m.addConstr(d == d.T)
        m.addConstr(d <= 2 / dim)
        for i in range(dim):
            m.addConstr(d[i][i] == 0)
            for j in range(i + 1, dim):
                m.addConstr(d[i][j] >= 1 / (3 * dim))
        m.setObjective(quicksum(d[i] @ c[i] for i in range(dim)), sense=GRB.MINIMIZE)
        m.optimize()
        return d.x

    @staticmethod
    def generate_from_graph(dim, noise_rate=0.01):
        nodes = [i for i in range(dim)]
        S, T = set(nodes), set()

        # Pick a random node, and mark it as visited and the current node.
        current_node = random.sample(S, 1).pop()
        S.remove(current_node)
        T.add(current_node)

        graph = nx.Graph()
        graph.add_nodes_from(nodes)

        # Create a random connected graph.
        while S:
            # Randomly pick the next node from the neighbors of the current node.
            # As we are generating a connected graph, we assume a complete graph.
            neighbor_node = random.sample(nodes, 1).pop()
            # If the new node hasn't been visited, add the edge from current to new.
            if neighbor_node not in T:
                edge = (current_node, neighbor_node)
                graph.add_edge(*edge)
                S.remove(neighbor_node)
                T.add(neighbor_node)
            # Set the new node as the current node.
            current_node = neighbor_node

        # Add random edges until the number of desired edges is reached.
        i = 0
        while i < dim / 3:
            edge = tuple(np.random.choice(nodes, size=2))
            if not graph.has_edge(*edge) and edge[0] != edge[1]:
                graph.add_edge(*edge)
                i += 1

        tau = nx.floyd_warshall_numpy(graph)
        entries = [(i, j) for i in range(1, dim) for j in range(i, dim)]
        idxs = random.sample(entries, max(1, int(noise_rate * len(entries))))
        for idx in idxs:
            tau[idx] = np.random.choice(range(2, dim))

        return tau / np.max(tau)

    @staticmethod
    def generate_from_grid(dim):
        points = np.array(random.sample([(i, j) for i in range(dim) for j in range(dim)], dim))

        d = np.zeros((dim, dim))
        for i in range(dim):
            d[i, :] = np.sqrt(np.sum((points[i] - points) ** 2, axis=1))
        return d / np.max(d)
