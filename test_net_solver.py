import json
import time

import networkx as nx
import numpy as np

from Data_.Datasets.bmep_dataset import BMEP_Dataset

from torch.utils.data import DataLoader

from Net.Nets.GNN1.gnn_1 import GNN_1
from Net.network_manager import NetworkManager
from Solvers.NetSolvers.heuristic_search import HeuristicSearch
from Solvers.NetSolvers.heuristic_search_2 import HeuristicSearch2
from Solvers.NetSolvers.net_solver import NetSolver
from Solvers.SWA.swa_solver import SwaSolver
from Solvers.UCTSolver.utc_solver import UtcSolver
from Solvers.solver import Solver

funs = Solver()
add_node = funs.add_node
compute_obj_val = funs.compute_obj_val_from_adj_mat

folder = 'GNN_TAU'
file = '_3.622'
data_folder = '03-M18_5_9'

net_manager = NetworkManager(folder, file)
dgn = net_manager.get_network()

data_ = BMEP_Dataset(folder_name=data_folder)

batch_size = 1000
start_test_set = net_manager.get_params()['train']['end']
n_test_problems = (data_.size - start_test_set) // 3
start_test_set = start_test_set + 3 - start_test_set % 3


r_swa_list = []
r_mcts_list = []
res_net_list = []
res_h_net_list = []

or_sol = []

for i in range(start_test_set, start_test_set + n_test_problems, 3):
    d = data_.d_mats[i]

    swa = SwaSolver(d.to('cpu').numpy())
    swa.solve()

    t = time.time()
    mcts_solver = UtcSolver(d.to('cpu').numpy())
    mcts_solver.solve(100)
    print('mcts ', time.time() - t)

    t = time.time()
    heuristic = HeuristicSearch2(d, dgn, width=15, distribution_runs=10)
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
    res_mcts = np.array_equal(mcts_solver.solution, sol)
    res_net = np.array_equal(net_solver.solution, sol)

    res_1 = np.array_equal(heuristic.solution, sol)
    r_swa_list.append(r_swa)
    r_mcts_list.append(res_mcts)
    res_net_list.append(res_net)
    res_h_net_list.append(res_1)

    or_sol.append(r_swa or res_1)

    print(i, "correct", r_swa, res_net, res_1, t,  t1, "    same sol ",
          np.array_equal(net_solver.solution, heuristic.solution), np.mean(or_sol))
    print(swa.obj_val, mcts_solver.obj_val, net_solver.obj_val, heuristic.obj_val, compute_obj_val(sol, d.to('cpu').numpy(), 6))
print("accuracy swa ", np.mean(r_swa_list))
print("accuracy mcts", np.mean(r_mcts_list))
print("accuracy ", np.mean(res_net_list))
print("accuracy", np.mean(res_h_net_list))
print('or sol', np.mean(or_sol))



