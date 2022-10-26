import time

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import string
from Instances.ip_solver import solve
from Pardi.pardi import Pardi
from Solvers.CPP.instance_cpp import InstanceCpp, CppSolver
from Solvers.IpSolver.ip_solver import IPSolver
from Solvers.PardiSolver.pardi_solver import PardiSolver
from Solvers.PardiSolver.pardi_solver_parallel import PardiSolverParallel
from Solvers.SWA.swa_solver import SwaSolver


def write_csv(d):
    n = d.shape[0]
    with open('test_input.txt', 'w', newline='\n') as csvfile:
        csvfile.write(str(n))
        csvfile.write('\n')
        for row in d:
            for el in row:
                csvfile.write(str(el))
                csvfile.write('\n')

class Instance:

    def __init__(self, d, labels=None, max_time=None, pardi_solver=False, log=False):

        self.d = self.sort_d(d)
        if log:
            write_csv(self.d)
        self.labels = [i for i in string.ascii_uppercase] if labels is None else labels
        n = self.d.shape[0]
        d_zeros = np.zeros((2*n - 2, 2*n - 2))
        d_zeros[:n, :n] = self.d
        # instance = Instance(self.d) if n < 7 else PardiSolverParallel(self.d)
        # self.out_time, self.obj_val, self.T = instance.solve()
        # self.T = self.T[:n, :n]
        instance = CppSolver(self.d)
        self.out_time = False
        self.obj_val, self.T = instance.solve(log)
        # instance = PardiSolver(self.d) if n < 7 else PardiSolverParallel(self.d)
        # out_time, obj_val, T = instance.solve()
        # T = T[:n, :n]
        # equal = np.array_equal(self.T, T)
        # if not equal:
        #     print("daniele")


        # if pardi_solver:
        #     instance = PardiSolver(self.d) if n < 7 else PardiSolverParallel(self.d)
        #     self.out_time, self.obj_val, self.T = instance.solve()
        #     self.T = self.T[:n, :n]
        # else:
        #     swa = SwaSolver(d_zeros)
        #     swa.solve()
        #     instance = IPSolver(self.d)
        #     self.out_time, self.obj_val, self.T = instance.solve(init_adj_sol=swa.solution, max_time=max_time)
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
