import random
import time
import warnings
from os import walk

import networkx as nx
import numpy as np
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceCalculator, DistanceMatrix

from FastME.fast_me import FastMeSolver
from Net.network_manager import NetworkManager
from Solvers.NJ.nj_phylo import NjPhylo
from Solvers.NJ.nj_solver import NjSolver
from Solvers.NetSolvers.heuristic_search_distribution import HeuristicSearchDistribution
from Solvers.PardiSolver.pardi_solver_parallel import PardiSolverParallel
from Solvers.SWA.swa_new import SwaSolverNew
from Solvers.SWA.swa_solver import SwaSolver
from Solvers.UCTSolver.utc_solver import UtcSolver

warnings.simplefilter("ignore")
random.seed(0)
np.random.seed(0)


path = 'Data_/csv_'
filenames = sorted(next(walk(path), (None, None, []))[2])

mats = []
for file in filenames:
    if file[-4:] == '.txt':
        mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))


m = mats[3]
dim_dataset = m.shape[0]

dim = 9
better = []
worse = []
for _ in range(10):
    idx = random.sample(range(dim_dataset), k=dim)
    d = m[idx, :][:, idx]

    t = time.time()
    pardi = PardiSolverParallel(d)
    pardi.solve()
    print(pardi.obj_val)
    t = time.time() - t
    print(t)

    pardi_lin = PardiSolverParallel(d, False)
    pardi_lin.solve()

    print(np.array_equal(pardi.solution, pardi_lin.solution))




#0.1069240103125
#7.917540788650513