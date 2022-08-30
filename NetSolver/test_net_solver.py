import json
import time

import numpy as np

from Data_.Dataset.bmep_dataset import BMEP_Dataset

from torch.utils.data import DataLoader

from Instances.instance import Instance
from Net.Nets.GNN.gnn import GNN
from Net.Nets.GNN1.gnn_1 import GNN_1
from NetSolver.net_solver import NetSolver

path = 'Net/Nets/GNN1/1661348420.9005842_0.627/'

m = np.loadtxt('mat_ok', dtype='int')
print(m)


with open(path + 'params.json', 'r') as json_file:
    params = json.load(json_file)
    print(params)

net_params = params
data_ = BMEP_Dataset()
batch_size = 1000
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)


dgn = GNN_1(net_params=net_params, network=path + "weights.pt")
res_list = []

for i in range(1):
    d = data_.d_mats[i*3]
    t1 = time.time()
    net_solver = NetSolver(d, dgn)
    net_solver.solve()
    t1 = time.time() - t1
    # print(net_solver.solution)

    t2 = time.time()
    instance = Instance(d.to("cpu").numpy()[:6, :6])
    t2 = time.time() - t2
    # print(instance.adj_mat_solution)
    # print(d[:6, :6])
    res = np.array_equal(net_solver.solution, instance.adj_mat_solution)
    res_list.append(res)

    print(i, "correct", res, instance.problem.status, t1, t2)

print("accuracy", np.mean(res_list))



