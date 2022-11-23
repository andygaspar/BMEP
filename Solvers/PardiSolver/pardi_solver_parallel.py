import copy
import multiprocessing as mp
import networkx as nx
import numpy as np

from Solvers.solver import Solver


class OpenProcess:

    def __init__(self, mat, step, num_steps, best_val):
        self.mat, self.step, self.num_steps, self.best_val = mat, step, num_steps, best_val


class PardiSolverParallel(Solver):

    def __init__(self, d, bme_obj_fun=True):
        super().__init__(d)
        self.n_taxa = d.shape[0]
        self.m = self.n_taxa + self.n_taxa - 2
        self.obj_fun = self.bme if bme_obj_fun else self.lin_bme
        self.steps = 0
        self.best_val = 10 ** 5
        self.solution = None


    def bme(self, Tau):
        return np.sum([self.d[i, j] / 2 ** (Tau[i, j]) for i in range(self.n_taxa) for j in range(self.n_taxa)])

    def lin_bme(self, Tau):
        return np.sum([self.d[i, j] / Tau[i, j] for i in range(self.n_taxa) for j in range(self.n_taxa) if i != j])

    def recursion(self, proc: OpenProcess):
        new_procs, best_sol = [], None
        selection = np.array(np.nonzero(np.triu(proc.mat))).T
        recursion_step = selection.shape[0]
        if recursion_step == proc.num_steps:
            obj_val = self.obj_fun(self.get_tau_(proc.mat, self.n_taxa))
            # print(obj_val)
            # print(obj_val, proc.best_val)
            if obj_val < proc.best_val:
                proc.best_val = obj_val
                best_sol = proc
        else:
            for i, idxs in enumerate(selection):
                new_mat = self.add_node(copy.deepcopy(proc.mat), idxs, proc.step)
                if recursion_step >= proc.num_steps - 2 or i == 0:
                    n_p, sol = self.recursion(OpenProcess(new_mat, proc.step + 1, proc.num_steps, proc.best_val))
                    new_procs += n_p
                    if sol is not None and sol.best_val < proc.best_val:
                        proc.best_val = sol.best_val
                        best_sol = sol
                else:
                    new_procs.append(OpenProcess(new_mat, proc.step + 1, proc.num_steps, proc.best_val))
        return new_procs, best_sol

    def solve(self):
        mat = np.zeros((self.m, self.m))
        for i in range(3):
            mat[i, self.n_taxa] = mat[self.n_taxa, i] = 1
        procs = ([OpenProcess(mat, step=3, num_steps=self.n_taxa + self.n_taxa - 3, best_val=self.best_val)])
        num_procs = mp.cpu_count()
        pool = mp.Pool(num_procs)
        while procs:
            results = pool.map(self.recursion, procs)
            procs = []
            for res in results:
                procs += res[0]
                if res[1] is not None and res[1].best_val < self.best_val:
                    self.best_val = res[1].best_val
                    self.solution = res[1]

            for op in procs:
                op.best_val = self.best_val
        pool.close()
        pool.join()
        # self.recursion(copy.deepcopy(mat))
        self.solution = self.solution.mat
        self.T = self.get_tau(self.solution)
        self.obj_val = self.best_val
        return False, self.obj_val, self.T

    def add_node(self, adj_mat, idxs, new_node_idx):
        adj_mat[idxs[0], idxs[1]] = adj_mat[idxs[1], idxs[0]] = 0  # detach selected
        adj_mat[idxs[0], self.n_taxa + new_node_idx - 2] = adj_mat[
            self.n_taxa + new_node_idx - 2, idxs[0]] = 1  # reattach selected to new
        adj_mat[idxs[1], self.n_taxa + new_node_idx - 2] = adj_mat[
            self.n_taxa + new_node_idx - 2, idxs[1]] = 1  # reattach selected to new
        adj_mat[new_node_idx, self.n_taxa + new_node_idx - 2] = adj_mat[
            self.n_taxa + new_node_idx - 2, new_node_idx] = 1  # attach new

        return adj_mat
