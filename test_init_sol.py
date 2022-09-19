import json
import time

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from Data_.Dataset.bmep_dataset import BMEP_Dataset

from torch.utils.data import DataLoader

from Net.Nets.GNN1.gnn_1 import GNN_1
from Solvers.IpSolver.ip_solver import IPSolver
from Solvers.NJ.nj_solver import NjSolver
from Solvers.NetSolvers.heuristic_search import HeuristicSearch
from Solvers.NetSolvers.net_solver import NetSolver
from Solvers.SWA.swa_solver import SwaSolver
from Solvers.solver import Solver

funs = Solver()
add_node = funs.add_node
compute_obj_val = funs.compute_obj_val_from_adj_mat
logs = False

path = 'Net/Nets/GNN1/_3.645/'

with open(path + 'params.json', 'r') as json_file:
    params = json.load(json_file)
    print(params)

net_params = params
data_ = BMEP_Dataset()
batch_size = 1000
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)

dgn = GNN_1(net_params=net_params, network=path + "weights.pt")
r_swa_list = []
res_list = []
res_1_list = []
r_nj_list = []
tot = []

runs = 100

times = np.zeros((runs, 2))

for i in range(runs):
    d = data_.d_mats[i * 3]

    nj = NjSolver(d.to('cpu').numpy())
    nj.solve()

    swa = SwaSolver(d.to('cpu').numpy())
    swa.solve()

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
    pre_final_adj_mat = data_.adj_mats[i * 3 + 2].to("cpu").numpy()
    last_move = np.nonzero(data_.y[i * 3 + 2].view(10, 10).to("cpu").numpy())
    sol = add_node(pre_final_adj_mat, last_move, 5, 6)

    # pardi = PardiSolver(d.to("cpu").numpy()[:6, :6])
    # pardi.solve()
    instance = IPSolver(d.to("cpu").numpy()[:6, :6])
    instance.solve(init_adj_sol=swa.solution, logs=logs)
    times[i, 0] = instance.sol_time

    instance = IPSolver(d.to("cpu").numpy()[:6, :6])
    init_sol = swa.solution if swa.obj_val < heuristic.obj_val else heuristic.solution
    instance.solve(init_adj_sol=init_sol, logs=logs)
    times[i, 1] = instance.sol_time

    # instance_s = IPSolver(d.to("cpu").numpy()[:6, :6])
    # instance_s.solve(init_adj_sol=swa.solution, logs=logs)
    # print(instance_s.obj_val)

    r_nj = np.array_equal(nj.solution, sol)
    r_swa = np.array_equal(swa.solution, sol)
    res = np.array_equal(net_solver.solution, sol)
    res_1 = np.array_equal(heuristic.solution, sol)
    r_nj_list.append(r_nj)
    r_swa_list.append(r_swa)
    res_list.append(res)
    res_1_list.append(res_1)

    tot.append(r_swa or res_1)

    # print('h_s t', instance.sol_time, instance_s.sol_time, 'h s correct', res_1, r_swa, r_nj, 'objval', heuristic.obj_val, swa.obj_val)
    print('h s correct', res_1, r_swa, r_nj, tot[-1], 'objval', heuristic.obj_val, swa.obj_val)

    # print(i, "correct", r_swa, res, res_1, t,  t1, "    same sol ", np.array_equal(net_solver.solution, heuristic.solution))
    # print(swa.obj_val, net_solver.obj_val, heuristic.obj_val, compute_obj_val(sol, d.to('cpu').numpy(), 6))

print("swa", np.mean(r_swa_list))
print("nj", np.mean(r_nj_list))
print("net", np.mean(res_list))
print("heu", np.mean(res_1_list))
print('tot', np.mean(tot))
plt.rcParams['figure.figsize'] = (20, 10)

plt.plot(range(runs), times[:, 0], label='swa')
plt.plot(range(runs), times[:, 1], label='swa + nn')
plt.legend()
plt.ylabel('time in secs')
plt.xlabel('run')
plt.tight_layout()
plt.show()


