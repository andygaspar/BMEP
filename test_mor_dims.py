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
from Solvers.IpSolver.ip_solver import IPSolver
from Solvers.NJ.nj_solver import NjSolver
from Solvers.NetSolver.heuristic_search import HeuristicSearch
from Solvers.SWA.swa_solver import SwaSolver

warnings.simplefilter("ignore")

path = 'Data_/csv_'
filenames = sorted(next(walk(path), (None, None, []))[2])

mats = []
for file in filenames:
    mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))


m = mats[0]
# random.seed(0)
path = 'Net/Nets/GNN1/_3.645/'

with open(path + 'params.json', 'r') as json_file:
    params = json.load(json_file)
    print(params)

net_params = params

dgn = GNN_1(net_params=net_params, network=path + "weights.pt")

dim = 9
better = []

for _ in range(100):
    idx = random.sample(range(12), k=dim)
    d = np.zeros((dim*2-2, dim*2-2))
    d[:dim, :dim] = m[idx, :][:, idx]


    nj = NjSolver(d)
    nj.solve()
    print(nj.obj_val)

    swa = SwaSolver(d)
    swa.solve()
    print(swa.obj_val)

    t = time.time()
    heuristic = HeuristicSearch(torch.tensor(d).to(torch.float).to("cuda:0"), dgn, 4)
    heuristic.solve()
    print(heuristic.obj_val)
    print('')
    t = time.time() - t

    better.append(swa.obj_val >= heuristic.obj_val)

print(np.mean(better))
# instance = IPSolver(d[:dim, :dim])
# instance.solve(init_adj_sol=swa.solution, logs=True)