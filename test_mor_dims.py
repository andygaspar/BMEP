import copy
import json
import random
import time
import warnings
from os import walk
import numpy as np
import torch
from Data_.Datasets.bmep_dataset import BMEP_Dataset
from FastME.fast_me import FastMeSolver
from Net.network_manager import NetworkManager
from Solvers.BBSolver.bb_solver import BB_Solver
from Solvers.IpSolver.ip_solver import IPSolver
from Solvers.NJ.nj_solver import NjSolver
from Solvers.NetSolvers.heuristic_search import HeuristicSearch
from Solvers.NetSolvers.heuristic_search_2 import HeuristicSearch2
from Solvers.NetSolvers.heuristic_search_3 import HeuristicSearch3
from Solvers.NetSolvers.net_solver import NetSolver
from Solvers.PardiSolver.pardi_solver import PardiSolver
from Solvers.PardiSolver.pardi_solver_parallel import PardiSolverParallel
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
# random.seed(0)
supervised = False
folder = 'GNN_TAU'
# file = '_3.622'
file = '_0.007_0'


# data_folder = '6_taxa_0'

net_manager = NetworkManager(folder, file, supervised=supervised)
dgn = net_manager.get_network()

dim = 12
better = []
worse = []
for _ in range(100):
    idx = random.sample(range(dim_dataset), k=dim)
    d = m[idx, :][:, idx]

    nj = NjSolver(d)
    nj.solve()

    swa = SwaSolver(d)
    swa.solve()

    t = time.time()
    heuristic = HeuristicSearch(d, dgn, width=10)
    heuristic.solve()
    t1 = time.time() - t

    print('heur time ', time.time() - t)

    t = time.time()
    mcts_solver = UtcSolver(d)
    mcts_solver.solve(20)
    print('mcts ', time.time() - t)

    fast = FastMeSolver(d)
    fast.solve()

    # t1 = time.time()
    # heuristic_ = HeuristicSearch3(torch.tensor(d).to(torch.float).to("cuda:0"), dgn, width=30, distribution_runs=300)
    # heuristic_.solve()
    # # print(heuristic_.obj_val)
    # t1 = time.time() - t1

    # t = time.time()
    # bb_solver = BB_Solver(d)
    # bb_solver.solve()
    # print(bb_solver.nodes)
    # print('done', time.time() - t)

    # t = time.time()
    # heuristic = HeuristicSearch(torch.tensor(d).to(torch.float).to("cuda:0"), dgn, 4)
    # heuristic.solve()
    # print(heuristic.obj_val)
    # print('')
    # t = time.time() - t

    # pardi = PardiSolverParallel(d[:dim, :dim])
    # pardi.solve()
    bet = fast.obj_val > mcts_solver.obj_val
    wor = fast.obj_val < mcts_solver.obj_val
    outcome = 'better' if bet else ('worse' if wor else 'equal')
    print(_, swa.obj_val, heuristic.obj_val, fast.obj_val,  mcts_solver.obj_val) #, pardi.obj_val, mcts_solver.obj_val == pardi.obj_val, outcome)

    better.append(bet)
    worse.append(wor)

print(np.mean(better))
print(np.mean(worse))

# instance = IPSolver(d[:dim, :dim])
# instance.solve(init_adj_sol=swa.solution, logs=True)

