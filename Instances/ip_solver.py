import numpy as np
import gurobipy as gb
from itertools import combinations


def solve(D, max_time):
    bmep = gb.Model()
    bmep.modelSense = gb.GRB.MINIMIZE
    bmep.setParam('OutputFlag', 0)
    if max_time is not None:
        bmep.setParam('TimeLimit', max_time)

    n = D.shape[0]
    TuV = range(2 * n)
    T = range(n)
    V = range(n, 2 * n)
    L = range(D.shape[0] - 1)

    x = bmep.addMVar((2 * n, 2 * n, n - 1), vtype=gb.GRB.BINARY)
    y = bmep.addMVar((2 * n, 2 * n, 2 * n, 2 * n), vtype=gb.GRB.BINARY)

    bmep.setObjective(
        gb.quicksum(
            (gb.quicksum(2 ** (-(k + 1)) * x[i, j, k] for k in L[1:])
             * D[i, j] for j in T for i in T if i != j))
    )

    # (32)
    combs = list(combinations(TuV, 2))
    for c in combs:
        i, j = c
        bmep.addConstr(
            gb.quicksum(x[i, j, k] for k in L) == 1
        )

    # (33)
    for k in L:
        for j in TuV:
            for i in range(j):
                bmep.addConstr(
                    x[j, i, k] == x[i, j, k]
                )

    # (34)
    for i in T:
        bmep.addConstr(
            gb.quicksum(x[i, j, k] * 2 ** (-(k + 1)) for j in T if i != j for k in L[1:]) == 1 / 2
        )

    # (35)
    bmep.addConstr(
        gb.quicksum(gb.quicksum(x[i, j, k] for i in T for j in T if i != j) * (k + 1) * 2 ** (-(k + 1)) for k in L[1:])
        == 2 * n - 3
    )

    # (36) (37)
    combs = list(combinations(TuV, 4))
    for c in combs:
        i, j, q, t = c
        bmep.addConstr(
            gb.quicksum((k + 1) * (x[i, j, k] + x[q, t, k]) for k in L) <=
            gb.quicksum((k + 1) * (x[i, q, k] + x[j, t, k]) for k in L) + (2 * n - 2) * y[i, j, q, t]

        )

        bmep.addConstr(
            gb.quicksum((k + 1) * (x[i, j, k] + x[q, t, k]) for k in L) <=
            gb.quicksum((k + 1) * (x[i, t, k] + x[j, q, k]) for k in L) +
            (2 * n - 2) * (1 - y[i, j, q, t])

        )

    # (38) (39)
    combs = list(combinations(T, 2))
    for c in combs:
        i, j = c
        bmep.addConstr(x[i, j, 0] == 0)

    for i in T:
        bmep.addConstr(
            gb.quicksum(x[i, j, 0] for j in V) == 1
        )

    # (40)
    for i in V:
        bmep.addConstr(
            gb.quicksum(x[i, j, 0] for j in TuV if i != j) == 3
        )

    # (41)
    combs = list(combinations(V, 3))
    for c in combs:
        i, j, l = c
        bmep.addConstr(
            x[i, j, 0] + x[i, l, 0] + x[l, j, 0] <= 2
        )

    # (42)
    combs = list(combinations(T, 2))
    for c in combs:
        i, j = c
        for l in V:
            for k in L[1:-1]:
                bmep.addConstr(
                    x[i, j, k] + 1 >= x[i, l, k - 1] + x[l, j, 0]
                )
    # (43)
    combs = list(combinations(TuV, 3))
    for c in combs:
        i, j, l = c
        for k in L[2:-1]:
            bmep.addConstr(
                x[i, j, k] + x[i, j, k - 2] + 1 >= x[i, l, k - 1] + x[l, j, 0]
            )

    bmep.optimize()
    print("status", bmep.status)
    out_time = bmep.status == 9

    if out_time:
        return out_time, None, None, None
    else:
        sol = np.zeros((n, n), dtype=int)

        for i in T:
            for j in T:
                for k in L:
                    if x.x[i, j, k] > 0.5:
                        sol[i, j] = k + 1

        return out_time, bmep, sol, bmep.objVal

