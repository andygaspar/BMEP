import numpy as np

from Data_.Dataset.bmep_dataset import BMEP_Dataset

from torch.utils.data import DataLoader

from Instances.instance import Instance
from Net.Nets.gnn import GNN
from NetSolver.net_solver import NetSolver

data_ = BMEP_Dataset()
batch_size = 128
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)


dgn = GNN(num_inputs=2, h_dimension=512, hidden_dim=512, num_messages=7, network="Net/Nets/net3.68.pt")
for i in range(30):
    d = data_.d_mats[i*3]
    net_solver = NetSolver(d, dgn)
    net_solver.solve()
    # print(net_solver.solution)

    instance = Instance(d.to("cpu").numpy()[:6, :6])
    # print(instance.adj_mat_solution)
    # print(d[:6, :6])

    print("correct", np.array_equal(net_solver.solution, instance.adj_mat_solution))


