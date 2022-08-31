import json
import time

import numpy as np

from Data_.Dataset.bmep_dataset import BMEP_Dataset

from torch.utils.data import DataLoader

from Instances.instance import Instance
from Net.Nets.GNN.gnn import GNN
from Net.Nets.GNN1.gnn_1 import GNN_1
from NetSolver.heuristic_search import HeuristicSearch
from NetSolver.net_solver import NetSolver
from NetSolver.old_solver import OldNetSolver

path = 'Net/Nets/GNN/_3.66/'


with open(path + 'params.json', 'r') as json_file:
    params = json.load(json_file)
    print(params)

net_params = params
data_ = BMEP_Dataset()
batch_size = 1000
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)


dgn = GNN(net_params=net_params, network=path + "weights.pt")
res_list = []
res_1_list = []

for i in range(20):
    d = data_.d_mats[i*3]
    t = time.time()
    heuristic = HeuristicSearch(d, dgn, 2)
    heuristic.solve()
    t = time.time() - t
    print(heuristic.solution)

    old_net_solver = OldNetSolver(d, dgn)
    old_net_solver.solve()

    t1 = time.time()
    net_solver = NetSolver(d, dgn)
    net_solver.solve()
    t1 = time.time() - t1
    print(net_solver.solution)
    print(data_.y[i * 3 + 2])

    # print(net_solver.solution)

    t2 = time.time()
    instance = Instance(d.to("cpu").numpy()[:6, :6])
    t2 = time.time() - t2
    # print(instance.adj_mat_solution)
    # print(d[:6, :6])
    r = np.array_equal(old_net_solver.solution, instance.adj_mat_solution)
    res_1 = np.array_equal(heuristic.solution, instance.adj_mat_solution)
    res = np.array_equal(net_solver.solution, instance.adj_mat_solution)
    res_list.append(res)
    res_1_list.append(res_1)

    print(np.array_equal(data_.y[i*3 + 2].to("cpu").numpy(), instance.adj_mat_solution))
    print(i, "old", r)
    print(i, "correct", res, instance.problem.status, t1, t2)
    print(i, "correct", res_1, instance.problem.status, t, t2, '\n')

print("accuracy", np.mean(res_list))
print("accuracy", np.mean(res_1_list))



