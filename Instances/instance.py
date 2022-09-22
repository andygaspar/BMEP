import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import string
from Instances.ip_solver import solve
from Pardi.pardi import Pardi
from Solvers.IpSolver.ip_solver import IPSolver
from Solvers.SWA.swa_solver import SwaSolver


class Instance:

    def __init__(self, d, labels=None, max_time=None):
        self.d = self.sort_d(d)
        self.labels = [i for i in string.ascii_uppercase] if labels is None else labels
        n = self.d.shape[0]
        d_zeros = np.zeros((2*n - 2, 2*n - 2))
        d_zeros[:n, :n] = self.d
        swa = SwaSolver(d_zeros)
        swa.solve()
        instance = IPSolver(self.d)
        self.out_time, self.problem, self.obj_val, self.T = instance.solve(init_adj_sol=swa.solution, max_time=max_time)
        # self.out_time, self.problem, self.obj_val, self.T = self.solve(max_time)
        if not self.out_time:
            self.pardi = Pardi(self.T)
            self.graph = self.pardi.get_graph()
            self.adj_mats = self.pardi.adj_mats[:-1]
            self.masks = [np.triu(mat) for mat in self.adj_mats]
            self.results = self.set_results(self.pardi.adj_mats)
            self.adj_mat_solution = self.pardi.adj_mats[-1]

    @staticmethod
    def sort_d(d):
        dist_sum = np.sum(d, axis=0)
        order = np.argsort(dist_sum)
        sorted_d = np.zeros_like(d)
        for i in order:
            for j in order:
                sorted_d[i, j] = d[order[i], order[j]]
        return sorted_d

    def solve(self, max_time):
        out_time, problem, T, obj_val = solve(self.d, max_time)
        return out_time, problem, obj_val, T

    def print_graph(self):
        nx.draw(self.graph, node_color=[self.graph.nodes[node]["color"] for node in self.graph.nodes],
                with_labels=True, font_weight='bold')
        plt.show()

    @staticmethod
    def set_results(adj_mats):
        results = []
        for i in range(len(adj_mats)-1):
            result = np.triu(adj_mats[i] - adj_mats[i+1])
            result[result < 0] = 0
            results.append(result)
        return results
