from os import walk
import numpy as np
import gurobipy as gb

path = 'Data_/csv_'
filenames = sorted(next(walk(path), (None, None, []))[2])

mats = []
for file in filenames:
    mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))

D = mats[0]

bmep = gb.Model()
bmep.modelSense = gb.GRB.MINIMIZE
bmep.setParam('OutputFlag', 0)

n = D.shape[0]
N = range(n)
L = range(D.shape[0] - 1)

x = bmep.addMVar((n, n, n - 1), vtype=gb.GRB.BINARY)
y = bmep.addMVar((n, n, n, n), vtype=gb.GRB.BINARY)

for i in N:
    for j in N:
        if i != j:
            bmep.addConstr(
                gb.quicksum(x[i, j, k] for k in L) == 1
            )

for k in L:
    for j in N:
        for i in range(j):
            bmep.addConstr(
                x[i, j, k] == x[i, j, k]
            )

for i in N:
    bmep.addConstr(
        gb.quicksum(x[i, j, k] * 2**(-k) for j in N if i != j for k in L[1:]) == 1/2
    )


bmep.addConstr(
    gb.quicksum(x[i, j, k] * k * 2**(-k) for i in N for j in N if i != j for k in L[1:]) == 2 * n - 3
)


bmep.setObjective(
    gb.quicksum(
        (gb.quicksum(2 ** (-k) * x[i, j, k] for k in L)
         * D[i, j] for j in N for i in N if i != j))
)
bmep.optimize()  # equivalent to solve() for xpress

print(x.x)
print("optimal" if bmep.status == 2 else ("infeasible" if bmep.status == 3 else (
    "unbounded" if bmep.status == 5 else "check the link page for other status codes")))
