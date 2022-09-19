import copy
import json
import random
import time
import warnings
from os import walk

import numpy as np
import torch

from Instances.generator import Generator
from Instances.instance import Instance
from Net.Nets.GNN1.gnn_1 import GNN_1
from Net.Nets.GNN_TAU.gnn_tau import GNN_TAU
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
file = '_4.045'

net_manager = NetworkManager()
dgn = net_manager.get_network(folder, file)

dim = 7
better = []

for _ in range(100):
    idx = random.sample(range(12), k=dim)
    d = np.zeros((dim*2-2, dim*2-2))
    d[:dim, :dim] = m[idx, :][:, idx]


    nj = NjSolver(d)
    nj.solve()

    swa = SwaSolver(d)
    swa.solve()
    print(swa.solution)
    print(swa.compute_obj_val_from_adj_mat(swa.solution[:7, :7], d[:7, :7], 7))

    t1 = time.time()
    heuristic = HeuristicSearch2(torch.tensor(d).to(torch.float).to("cuda:0"), dgn, 50)
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