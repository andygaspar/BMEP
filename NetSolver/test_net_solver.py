import time

import numpy as np

from Data_.Dataset.bmep_dataset import BMEP_Dataset

from torch.utils.data import DataLoader

from Instances.instance import Instance
from Net.Nets.gnn import GNN
from NetSolver.net_solver import NetSolver

data_ = BMEP_Dataset()
batch_size = 128
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)


dgn = GNN(num_inputs=2, h_dimension=512, hidden_dim=512, num_messages=4, network="Net/Nets/net3.68.pt")
res_list = []
for i in range(200):
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

