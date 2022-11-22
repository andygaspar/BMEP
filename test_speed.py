import random
import time
import warnings
from os import walk
import numpy as np
from FastME.fast_me import FastMeSolver
from Net.network_manager import NetworkManager
from Solvers.NJ.nj_solver import NjSolver
from Solvers.NetSolvers.heuristic_search_distribution import HeuristicSearchDistribution
from Solvers.SWA.swa_solver import SwaSolver
from Solvers.UCTSolver.utc_solver import UtcSolver


warnings.simplefilter("ignore")


path = 'Data_/csv_'
filenames = sorted(next(walk(path), (None, None, []))[2])

mats = []
for file in filenames:
    if file[-4:] == '.txt':
        mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))


m = mats[3]
dim_dataset = m.shape[0]

dim = 50

runs = 100

results = np.zeros((runs, 4))


for run in range(runs):
    idx = random.sample(range(dim_dataset), k=dim)
    d = m[idx, :][:, idx]

    t = time.time()
    swa = SwaSolver(d)
    swa.solve()
    t = time.time() - t

    t1 = time.time()
    mcts = UtcSolver(d)
    mcts.solve(10)
    t1 = time.time() - t1

    t2 = time.time()
    fast = FastMeSolver(d)
    fast.solve(bme=False, nni=False, triangular_inequality=False, logs=False)
    t2 = time.time() - t2
    print("ffff")
    t3 = time.time()
    fast1 = FastMeSolver(d)
    fast1.solve(bme=True, nni=False, triangular_inequality=False, logs=False)
    t3 = time.time() - t3

    t4 = time.time()
    fast2 = FastMeSolver(d)
    fast2.solve(bme=True, nni=True, triangular_inequality=False, logs=False)
    t4 = time.time() - t4

    t5 = time.time()
    fast3 = FastMeSolver(d)
    fast3.solve(bme=True, nni=True, digits=17, triangular_inequality=True, logs=False)
    t5 = time.time() - t5

    order = list(np.argsort([fast.obj_val, fast1.obj_val, fast2.obj_val, fast3.obj_val]))
    score = [order.index(i) for i in range(4)]
    results[run] = score

    print(t1, t2, t3, t4, t5)
    print(mcts.obj_val, fast.obj_val, fast1.obj_val, fast2.obj_val, fast3.obj_val, "\n")

    # print(t, t1, t2, t3, swa.obj_val, utc.obj_val, "f1", fast.obj_val, "f2", fast1.obj_val)

print("results")
print(results.sum(axis=0))
print([results[:, i][results[:, i] == 0].shape[0] for i in range(4)])

