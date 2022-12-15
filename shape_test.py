import numpy as np

from Solvers.solver import Solver
import gurobipy as gb


class ShapeSolver(Solver):

    def __init__(self, d, adj_mat, Tau, relaxation=False):
        super(ShapeSolver, self).__init__(d)

        self.bmep = gb.Model()
        self.bmep.modelSense = gb.GRB.MINIMIZE
        self.bmep.setParam('OutputFlag', 0)
        self.adj_mat = adj_mat
        self.Tau = Tau

        self.x = self.bmep.addMVar((self.n_taxa, self.n_taxa), vtype=gb.GRB.BINARY)



    def set_objective(self):
        self.bmep.setObjective(
                gb.quicksum((2. ** (-self.Tau[j]) * self.d[i]).sum() * self.x[i, j]
                  for j in range(self.n_taxa) for i in range(self.n_taxa)
            )
        )

    def set_constraints(self):
        for i in range(self.n_taxa):
            self.bmep.addConstr(
                 gb.quicksum(self.x[i, j] for j in range(self.n_taxa)) == 1
            )

            self.bmep.addConstr(
                gb.quicksum(self.x[j, i] for j in range(self.n_taxa)) == 1
            )
    def solve(self):
        self.set_objective()
        self.set_constraints()
        self.bmep.optimize()
        self.obj_val = self.bmep.objVal