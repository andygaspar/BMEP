import time

import networkx as nx
import numpy as np
import gurobipy as gb
from itertools import combinations, permutations

from Solvers.solver import Solver


class IPSolver(Solver):

    def __init__(self, d, relaxation=False):
        super(IPSolver, self).__init__(d)

        self.bmep = gb.Model()
        self.bmep.modelSense = gb.GRB.MINIMIZE

        self.TuV = range(self.m)
        self.T = range(self.n_taxa)
        self.V = range(self.n_taxa, self.m)
        self.L = range(self.n_taxa - 1)

        var_type = gb.GRB.BINARY if not relaxation else gb.GRB.CONTINUOUS

        self.x = self.bmep.addMVar((self.m, self.m, self.n_taxa - 1), vtype=var_type)
        self.y = self.bmep.addMVar((self.m, self.m, self.m, self.m), vtype=var_type)
        self.sol_time = None

    def set_objective(self):
        self.bmep.setObjective(
            gb.quicksum(
                (gb.quicksum(2 ** (-(k + 1)) * self.x[i, j, k] for k in self.L[1:])
                 * self.d[i, j] for j in self.T for i in self.T if i != j))
        )

    def solve(self, max_time=None, init_adj_sol=None, logs=False, print_time=False):

        if not logs:
            self.bmep.setParam('OutputFlag', 0)

        if init_adj_sol is not None:
            self.init_sol(init_adj_sol)

        if max_time is not None:
            self.bmep.setParam('TimeLimit', max_time)

        self.set_objective()
        t = time.time()
        self.set_constraints()
        if print_time:
            print('constr settings ', time.time() - t)

        t = time.time()
        self.bmep.optimize()
        self.sol_time = time.time() - t
        if print_time:
            print('sol time ', self.sol_time)

        # print("status", self.bmep.status)
        out_time = self.bmep.status == 9

        if out_time:
            return out_time, None, None, None
        else:
            T_sol = np.zeros((self.n_taxa, self.n_taxa), dtype=int)

            for i in self.T:
                for j in self.T:
                    for k in self.L:
                        if self.x.x[i, j, k] > 0.5:
                            T_sol[i, j] = k + 1

            self.obj_val = self.bmep.objVal
            return out_time, self.obj_val, T_sol

    def get_lp(self):
        self.set_objective()
        self.set_constraints()
        return self.bmep.relax()

    def set_constraints(self):

        # (32)
        combs = list(combinations(self.TuV, 2))
        for c in combs:
            i, j = c
            self.bmep.addConstr(
                gb.quicksum(self.x[i, j, k] for k in self.L) == 1
            )

        # (33)
        for k in self.L:
            for j in self.TuV:
                for i in range(j):
                    self.bmep.addConstr(
                        self.x[j, i, k] == self.x[i, j, k]
                    )

        # (34)
        for i in self.T:
            self.bmep.addConstr(
                gb.quicksum(self.x[i, j, k] * 2 ** (-(k + 1)) for j in self.T if i != j for k in self.L[1:]) == 1 / 2
            )

        # (35)
        self.bmep.addConstr(
            gb.quicksum(gb.quicksum(self.x[i, j, k] for i in self.T for j in self.T if i != j)
                        * (k + 1) * 2 ** (-(k + 1)) for k in self.L[1:])
            == 2 * self.n_taxa - 3
        )

        # (36) (37)
        combs = list(combinations(self.TuV, 4))
        for c in combs:
            for p in list(permutations(c)):
                i, j, q, t = p
                self.bmep.addConstr(
                    gb.quicksum((k + 1) * (self.x[i, j, k] + self.x[q, t, k]) for k in self.L) <=
                    gb.quicksum((k + 1) * (self.x[i, q, k] + self.x[j, t, k]) for k in self.L)
                    + (self.m) * self.y[i, j, q, t]

                )

                self.bmep.addConstr(
                    gb.quicksum((k + 1) * (self.x[i, j, k] + self.x[q, t, k]) for k in self.L) <=
                    gb.quicksum((k + 1) * (self.x[i, t, k] + self.x[j, q, k]) for k in self.L) +
                    (self.m) * (1 - self.y[i, j, q, t])

                )

        # (38) (39)
        combs = list(combinations(self.T, 2))
        for c in combs:
            i, j = c
            self.bmep.addConstr(self.x[i, j, 0] == 0)

        for i in self.T:
            self.bmep.addConstr(
                gb.quicksum(self.x[i, j, 0] for j in self.V) == 1
            )

        # (40)
        for i in self.V:
            self.bmep.addConstr(
                gb.quicksum(self.x[i, j, 0] for j in self.TuV if i != j) == 3
            )

        # (41)
        combs = list(combinations(self.V, 3))
        for c in combs:
            i, j, l = c
            self.bmep.addConstr(
                self.x[i, j, 0] + self.x[i, l, 0] + self.x[l, j, 0] <= 2
            )

        # (42)
        combs = list(combinations(self.T, 2))
        for c in combs:
            i, j = c
            for l in self.V:
                for k in self.L[1:-1]:
                    self.bmep.addConstr(
                        self.x[i, j, k] + 1 >= self.x[i, l, k - 1] + self.x[l, j, 0]
                    )
        # (43)
        combs = list(combinations(self.TuV, 3))
        for c in combs:
            i, j, l = c
            for k in self.L[2:-1]:
                self.bmep.addConstr(
                    self.x[i, j, k] + self.x[i, j, k - 2] + 1 >= self.x[i, l, k - 1] + self.x[l, j, 0]
                )

    def init_sol(self, adj_sol):
        for i in range(self.bmep.NumStart):
            print(22)
        g = nx.from_numpy_matrix(adj_sol)
        Tau = nx.floyd_warshall_numpy(g)
        for i in self.TuV:
            for j in self.TuV:
                for k in range(self.n_taxa - 1):
                    self.x[i, j, k].Start = 1 if k == Tau[i, j] - 1 else 0

                    for q in self.TuV:
                        for t in self.TuV:
                            self.y[i, j, q, t].Start = 1 if Tau[i, t] + Tau[j, q] >= Tau[i, q] + Tau[
                                j, t] and i != j != q != t else 0


