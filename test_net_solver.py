import random
import time

import numpy as np

from Data_.Datasets.bmep_dataset import BMEP_Dataset

from Solvers.FastME.fast_me import FastMeSolver
from ML.Net.network_manager import NetworkManager
from Solvers.NetSolvers.heuristic_search_distribution import HeuristicSearchDistribution
from Solvers.NetSolvers.heuristic_search_dropout import HeuristicSearchDropOut
from Solvers.NetSolvers.net_solver import NetSolver
from Solvers.SWA.swa_solver import SwaSolver
from Solvers.solver import Solver

funs = Solver()
add_node = funs.add_node
compute_obj_val = funs.compute_obj_val_from_adj_mat

random.seed(0)
np.random.seed(0)

folder = 'EGAT_RL'
file = '_-245901.018_0'
data_folder = '03-M18_5_9' #'6_taxa_0'
n_test_problems = 100

net_manager = NetworkManager(folder, file=file, supervised=False)
dgn = net_manager.get_network()

# problems = torch.tensor([i for i in range(10000//3 + 1) for j in range(3)][:-2])
# torch.save(problems, 'Data_/Datasets/6_taxa_0/problems.pt')

start_test_set = net_manager.get_params()['train']['end']
start_test_set = 0
data_ = BMEP_Dataset(folder_name=data_folder, start=start_test_set)
problems = np.unique(data_.problems.numpy())[1:-1]

r_swa_list = []
res_list = []
res_1_list = []
or_sol = []
better = []
res_fast = []

for _ in range(n_test_problems):
    n = 0
    # while n != 7 and n != 8:
    pb = random.choice(problems)
    pb_idxs = (data_.problems == pb).nonzero().flatten()
    d = data_.d_mats[pb_idxs[0]]
    n = 3 + pb_idxs.shape[0]
    m = n * 2 - 2
    d = d[: n, : n]

    swa = SwaSolver(d.to('cpu').numpy())
    swa.solve()

    t1 = time.time()
    net_solver = NetSolver(d, dgn)
    net_solver.solve()
    t1 = time.time() - t1

    t = time.time()
    heuristic_distribution = HeuristicSearchDistribution(d, dgn, width=20)
    heuristic_distribution.solve()
    t = time.time() - t

    t = time.time()
    heuristic_drop = HeuristicSearchDropOut(d, dgn, width=20, distribution_runs=30)
    heuristic_drop.solve()
    t = time.time() - t
    # print(heuristic_drop.solution)


    # print(net_solver.solution)
    # print(data_.y[i * 3 + 2])

    # print(net_solver.solution)
    pre_final_adj_mat = data_.adj_mats[pb_idxs[-1]].to("cpu").numpy()
    size = int(data_.y[pb_idxs[-1]].shape[0]**(1/2))
    last_move = np.nonzero(data_.y[pb_idxs[-1]].view(size, size)[: m, : m].to("cpu").numpy())
    sol = add_node(pre_final_adj_mat, last_move, n-1, n)[:m, :m]

    fast = FastMeSolver(d.to("cpu").numpy())
    fast.solve()

    # t2 = time.time()
    # instance = Instance(d.to("cpu").numpy()[:6, :6])
    # t2 = time.time() - t2
    # print(instance.adj_mat_solution)
    # print(d[:6, :6])
    r_fast = fast.obj_val > compute_obj_val(sol, d.to('cpu').numpy(), n)
    r_swa = np.array_equal(swa.solution, sol)
    res = np.array_equal(net_solver.solution, sol)
    res_1 = np.array_equal(heuristic_drop.solution, sol)
    res_fast.append(r_fast)
    r_swa_list.append(r_swa)
    res_list.append(res)
    res_1_list.append(res_1)

    or_sol.append(r_swa or res_1)
    better.append(heuristic_drop.obj_val <= swa.obj_val)

    print(_, n, "correct", r_swa, res, res_1, r_fast, "    same sol ",
          np.array_equal(net_solver.solution, heuristic_drop.solution), np.mean(or_sol))
    print(swa.obj_val, net_solver.obj_val, heuristic_drop.obj_val, heuristic_distribution.obj_val,
          "f", fast.obj_val, compute_obj_val(sol, d.to('cpu').numpy(), n))
print("accuracy swa", np.mean(r_swa_list))
print("accuracy net", np.mean(res_list))
print("accuracy heuristic_drop", np.mean(res_1_list))
print("accuracy fasta", np.mean(res_fast))
print('or sol', np.mean(or_sol))
print('better', np.mean(better))



