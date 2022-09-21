import copy
import json
import random
import time
import warnings
from os import walk
import numpy as np
import torch
from Data_.Dataset.bmep_dataset import BMEP_Dataset
from Net.network_manager import NetworkManager
from Solvers.IpSolver.ip_solver import IPSolver
from Solvers.NJ.nj_solver import NjSolver
from Solvers.NetSolvers.heuristic_search import HeuristicSearch
from Solvers.NetSolvers.heuristic_search_2 import HeuristicSearch2
from Solvers.SWA.swa_solver import SwaSolver

warnings.simplefilter("ignore")

path = 'Data_/csv_'
filenames = sorted(next(walk(path), (None, None, []))[2])

mats = []
for file in filenames:
    mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))


m = mats[0]
# random.seed(0)
folder = 'GNN_TAU'
file = '_3.622'

data_ = BMEP_Dataset()
normalisation_factor = data_.max_d_mat
print(normalisation_factor, "pppppppppp")

net_manager = NetworkManager(folder, file)
dgn = net_manager.get_network()

dim = 5
better = []

for _ in range(100):
    idx = random.sample(range(12), k=dim)
    d = np.zeros((dim*2-2, dim*2-2))
    d[:dim, :dim] = m[idx, :][:, idx]

    nj = NjSolver(d)
    nj.solve()

    swa = SwaSolver(d)
    swa.solve()

    t1 = time.time()
    heuristic = HeuristicSearch2(torch.tensor(d).to(torch.float).to("cuda:0"), dgn, width=10)
    heuristic.solve()
    t1 = time.time() - t1

    # t = time.time()
    # heuristic = HeuristicSearch(torch.tensor(d).to(torch.float).to("cuda:0"), dgn, 4)
    # heuristic.solve()
    # print(heuristic.obj_val)
    # print('')
    # t = time.time() - t
    print(nj.obj_val, swa.obj_val, heuristic.obj_val, swa.obj_val >= heuristic.obj_val)

    better.append(swa.obj_val >= heuristic.obj_val)

print(np.mean(better))
# instance = IPSolver(d[:dim, :dim])
# instance.solve(init_adj_sol=swa.solution, logs=True)

