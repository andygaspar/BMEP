import json
import random
import time

import networkx as nx
import numpy as np
import torch

from Data_.Datasets.bmep_dataset import BMEP_Dataset

from torch.utils.data import DataLoader

from Net.Nets.GNN1.gnn_1 import GNN_1
from Net.network_manager import NetworkManager
from Solvers.NetSolvers.heuristic_search import HeuristicSearch
from Solvers.NetSolvers.heuristic_search_2 import HeuristicSearch2
from Solvers.NetSolvers.net_solver import NetSolver
from Solvers.SWA.swa_solver import SwaSolver
from Solvers.solver import Solver

funs = Solver()
add_node = funs.add_node
compute_obj_val = funs.compute_obj_val_from_adj_mat

folder = 'GNN_TAU'
file = '_5.497_0'
data_folder = '03-M18_9_13' #'6_taxa_0'
n_test_problems = 100

net_manager = NetworkManager(folder, file)
dgn = net_manager.get_network()

# problems = torch.tensor([i for i in range(10000//3 + 1) for j in range(3)][:-2])
# torch.save(problems, 'Data_/Datasets/6_taxa_0/problems.pt')

start_test_set = net_manager.get_params()['train']['end']
data_ = BMEP_Dataset(folder_name=data_folder, start=start_test_set)
problems = np.unique(data_.problems.numpy())[1:-1]

r_swa_list = []
res_list = []
res_1_list = []
or_sol = []
better = []

for _ in range(n_test_problems):
    n = 0
    # while n != 7 and n != 8:
    pb = random.choice(problems)
    pb_idxs = (data_.problems == pb).nonzero().flatten()
    d = data_.d_mats[pb_idxs[0]]
    n = 3 + pb_idxs.shape[0]
    m = n * 2 - 2
    d = d[: m, : m]

    swa = SwaSolver(d.to('cpu').numpy())
    swa.solve()

    t = time.time()
    heuristic = HeuristicSearch2(d, dgn, width=10, distribution_runs=400)
    heuristic.solve()
    t = time.time() - t
    # print(heuristic.solution)

    t1 = time.time()
    net_solver = NetSolver(d, dgn)
    net_solver.solve()
    t1 = time.time() - t1
    # print(net_solver.solution)
    # print(data_.y[i * 3 + 2])

    # print(net_solver.solution)
    pre_final_adj_mat = data_.adj_mats[pb_idxs[-1]].to("cpu").numpy()
    size = int(data_.y[pb_idxs[-1]].shape[0]**(1/2))
    last_move = np.nonzero(data_.y[pb_idxs[-1]].view(size, size)[: m, : m].to("cpu").numpy())
    sol = add_node(pre_final_adj_mat, last_move, n-1, n)[:m, :m]

    # pardi = PardiSolver(d.to("cpu").numpy()[:6, :6])
    # pardi.solve()

    # t2 = time.time()
    # instance = Instance(d.to("cpu").numpy()[:6, :6])
    # t2 = time.time() - t2
    # print(instance.adj_mat_solution)
    # print(d[:6, :6])

    r_swa = np.array_equal(swa.solution, sol)
    res = np.array_equal(net_solver.solution, sol)
    res_1 = np.array_equal(heuristic.solution, sol)
    r_swa_list.append(r_swa)
    res_list.append(res)
    res_1_list.append(res_1)

    or_sol.append(r_swa or res_1)
    better.append(heuristic.obj_val <= swa.obj_val)

    print(_, n, "correct", r_swa, res, res_1, t,  t1, "    same sol ",
          np.array_equal(net_solver.solution, heuristic.solution), np.mean(or_sol))
    print(swa.obj_val, net_solver.obj_val, heuristic.obj_val, compute_obj_val(sol, d.to('cpu').numpy(), n))
print("accuracy", np.mean(r_swa_list))
print("accuracy", np.mean(res_list))
print("accuracy", np.mean(res_1_list))
print('or sol', np.mean(or_sol))
print('better', np.mean(better))



