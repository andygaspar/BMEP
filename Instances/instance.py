import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import string
from Instances.ip_solver import solve
from Pardi.pardi import Pardi


class Instance:

    def __init__(self, d, labels=None):
        self.d = self.sort_d(d)
        self.labels = [i for i in string.ascii_uppercase] if labels is None else labels
        self.problem, self.obj_val, self.T = self.solve()
        self.pardi = Pardi(self.T)
        self.graph = self.pardi.get_graph()
        self.adj_mats = self.pardi.adj_mats[:-1]
        self.masks = [np.triu(mat) for mat in self.adj_mats]
        self.results = self.set_results(self.pardi.adj_mats)

    @staticmethod
    def sort_d(d):
        dist_sum = np.sum(d, axis=0)
        order = np.argsort(dist_sum)
        sorted_d = np.zeros_like(d)
        for i in order:
            for j in order:
                sorted_d[i, j] = d[order[i], order[j]]
        return sorted_d

    def solve(self):
        problem, T = solve(self.d)
        obj_val = self.problem.getObjective().getValue()
        return problem, obj_val, T

    def print_graph(self):
        nx.draw(self.graph, node_color=[self.graph.nodes[node]["color"] for node in self.graph.nodes],
                with_labels=True, font_weight='bold')
        plt.show()

    @staticmethod
    def set_results(adj_mats):
        masks = []
        for i in range(len(adj_mats)-1):
            mask = np.triu(adj_mats[i] - adj_mats[i])
            mask[mask < 0] = 0
            masks.append(mask)
        return masks
