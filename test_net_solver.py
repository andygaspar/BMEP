import json
import time

import networkx as nx
import numpy as np

from Data_.Dataset.bmep_dataset import BMEP_Dataset

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
file = '_3.622'

net_manager = NetworkManager()
dgn = net_manager.get_network(folder, file)

data_ = BMEP_Dataset()

batch_size = 1000
start_test_set = net_manager.get_params(folder, file)['train']['end']
n_test_problems = (data_.size - start_test_set) // 3
start_test_set = start_test_set + 3 - start_test_set % 3


# dataloader = DataLoader(dataset=data_[start_test_set:], batch_size=batch_size, shuffle=True)


r_swa_list = []
res_list = []
res_1_list = []
or_sol = []

for i in range(start_test_set, start_test_set + n_test_problems, 3):
    d = data_.d_mats[i]

    swa = SwaSolver(d.to('cpu').numpy())
    swa.solve()

    t = time.time()
    heuristic = HeuristicSearch2(d, dgn, width=5, distribution_runs=100)
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
    pre_final_adj_mat = data_.adj_mats[i + 2].to("cpu").numpy()
    last_move = np.nonzero(data_.y[i + 2].view(10, 10).to("cpu").numpy())
    sol = add_node(pre_final_adj_mat, last_move, 5, 6)

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

    print(i, "correct", r_swa, res, res_1, t,  t1, "    same sol ",
          np.array_equal(net_solver.solution, heuristic.solution), np.mean(or_sol))
    print(swa.obj_val, net_solver.obj_val, heuristic.obj_val, compute_obj_val(sol, d.to('cpu').numpy(), 6))
print("accuracy", np.mean(r_swa_list))
print("accuracy", np.mean(res_list))
print("accuracy", np.mean(res_1_list))
print('or sol', np.mean(or_sol))



