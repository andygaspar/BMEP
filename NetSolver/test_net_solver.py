import json
import time

import networkx as nx
import numpy as np

from Data_.Dataset.bmep_dataset import BMEP_Dataset

from torch.utils.data import DataLoader

from Instances.instance import Instance
from Net.Nets.GNN.gnn import GNN
from Net.Nets.GNN1.gnn_1 import GNN_1
from Net.Nets.GNNGRU.gnn_gru import GNN_GRU
from Net.network_manager import NetworkManager

from NetSolver.heuristic_search import HeuristicSearch
from NetSolver.net_solver import NetSolver
from NetSolver.old_solver import OldNetSolver
from PardiSolver.pardi_solver import PardiSolver


def add_node(adj_mat, idxs, new_node_idx):
    adj_mat[idxs[0], idxs[1]] = adj_mat[idxs[1], idxs[0]] = 0  # detach selected
    adj_mat[idxs[0], 6 + new_node_idx - 2] = adj_mat[6 + new_node_idx - 2, idxs[0]] = 1  # reattach selected to new
    adj_mat[idxs[1], 6 + new_node_idx - 2] = adj_mat[6 + new_node_idx - 2, idxs[1]] = 1  # reattach selected to new
    adj_mat[new_node_idx, 6 + new_node_idx - 2] = adj_mat[6 + new_node_idx - 2, new_node_idx] = 1  # attach new

    return adj_mat

def compute_obj_val(d, adj_mat, n):
    g = nx.from_numpy_matrix(adj_mat)
    Tau = nx.floyd_warshall_numpy(g)[:n, :n]
    d = d.to("cpu").numpy()
    return np.sum([d[i, j] / 2 ** (Tau[i, j]) for i in range(n) for j in range(n)])


folder = 'GNNGRU'
file = '_0.228'

net_manager = NetworkManager()

dgn = net_manager.get_network(folder, file)
data_ = BMEP_Dataset()
batch_size = 1000
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)


res_list = []
res_1_list = []

for i in range(100):
    d = data_.d_mats[i*3]
    t = time.time()
    heuristic = HeuristicSearch(d, dgn, 6)
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
    sol = add_node(pre_final_adj_mat, last_move, 5)

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


    print(i, "correct", res, res_1, t,  t1, "    same sol ", np.array_equal(net_solver.solution, heuristic.solution), heuristic.prob)
    # print(compute_obj_val(d, net_solver.solution, 6), heuristic.solution_object.obj_val, compute_obj_val(d, sol, 6), pardi.best_val)
print("accuracy", np.mean(res_list))
print("accuracy", np.mean(res_1_list))



