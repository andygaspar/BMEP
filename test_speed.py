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


m = mats[2]
dim_dataset = m.shape[0]

dim = 18


for _ in range(1):
    idx = random.sample(range(dim_dataset), k=dim)
    d = m[idx, :][:, idx]
    t = time.time()
    swa = SwaSolver(d)
    swa.solve()
    t = time.time() - t

    t1 = time.time()
    utc = UtcSolver(d)
    utc.solve(10)
    t1 = time.time() - t1

    t2 = time.time()
    utc1 = UtcSolver(d)
    utc1.solve(100)
    t2 = time.time() - t2

    print(t, t1, t2, swa.obj_val, utc.obj_val, utc1.obj_val)