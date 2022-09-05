import json
import time

import networkx as nx
import numpy as np

from Data_.Dataset.bmep_dataset import BMEP_Dataset

from torch.utils.data import DataLoader

from Net.Nets.GNN1.gnn_1 import GNN_1
from Solvers.NetSolver.heuristic_search import HeuristicSearch
from Solvers.NetSolver.net_solver import NetSolver
from Solvers.solver import Solver

funs = Solver()
add_node = funs.add_node
compute_obj_val = funs.compute_obj_val_from_adj_mat

path = 'Net/Nets/GNN1/_3.645/'


with open(path + 'params.json', 'r') as json_file:
    params = json.load(json_file)
    print(params)

net_params = params
data_ = BMEP_Dataset()
batch_size = 1000
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)


dgn = GNN_1(net_params=net_params, network=path + "weights.pt")
res_list = []
res_1_list = []

for i in range(100):
    d = data_.d_mats[i*3]
    t = time.time()
    heuristic = HeuristicSearch(d, dgn, 10)
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
    pre_final_adj_mat = data_.adj_mats[i*3 + 2].to("cpu").numpy()
    last_move = np.nonzero(data_.y[i*3 + 2].view(10, 10).to("cpu").numpy())
    sol = add_node(pre_final_adj_mat, last_move, 5, 6)

    # pardi = PardiSolver(d.to("cpu").numpy()[:6, :6])
    # pardi.solve()

    # t2 = time.time()
    # instance = Instance(d.to("cpu").numpy()[:6, :6])
    # t2 = time.time() - t2
    # print(instance.adj_mat_solution)
    # print(d[:6, :6])

    res = np.array_equal(net_solver.solution, sol)
    res_1 = np.array_equal(heuristic.solution, sol)
    res_list.append(res)
    res_1_list.append(res_1)


    print(i, "correct", res, res_1, t,  t1, "    same sol ", np.array_equal(net_solver.solution, heuristic.solution))
    # print(compute_obj_val(d, net_solver.solution, 6), heuristic.solution_object.obj_val, compute_obj_val(d, sol, 6), pardi.best_val)
print("accuracy", np.mean(res_list))
print("accuracy", np.mean(res_1_list))



