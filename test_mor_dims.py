import random
import time
import warnings
from os import walk

import numpy as np

from FastME.fast_me import FastMeSolver
from Solvers.NJ.nj_phylo import NjPhylo
from Solvers.NJ.nj_solver import NjSolver
from Old.SWA_CPP.swa_new import SwaSolverNew
from Solvers.SWA.swa_solver import SwaSolver

warnings.simplefilter("ignore")
random.seed(0)
np.random.seed(0)


path = 'Data_/csv_'
filenames = sorted(next(walk(path), (None, None, []))[2])

mats = []
for file in filenames:
    if file[-4:] == '.txt':
        mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))


m = mats[3]
dim_dataset = m.shape[0]
# random.seed(0)
# supervised = True
# folder = 'GNN_TAU'
# file = '_3.622'
# file = '_0.092_0'
supervised = False
folder = 'EGAT_RL'
file = '_-245901.018_0'

# data_folder = '6_taxa_0'

# net_manager = NetworkManager(folder, file=file, supervised=supervised)
# dgn = net_manager.get_network()

dim = 40
better = []
worse = []
for _ in range(20):
    idx = random.sample(range(dim_dataset), k=dim)
    d = m[idx, :][:, idx]

    t = time.time()
    swa_new = SwaSolverNew(d)
    swa_new.solve()
    t = time.time() - t

    nj = NjSolver(d)
    nj.solve()

    nj_phylo = NjPhylo(d)
    nj_phylo.solve()
    # print(nj.obj_val, nj_phylo.obj_val)


    t1 = time.time()
    swa = SwaSolver(d)
    swa.solve()
    t1 = time.time() - t1

    print(t, t1)



    # t = time.time()
    # heuristic = HeuristicSearchDistribution(d, dgn, width=20
    #                                         )
    # heuristic.solve()
    # t1 = time.time() - t

    # print('heur time ', time.time() - t)

    # t = time.time()
    # mcts_solver = UtcSolver(d)
    # mcts_solver.solve(20)
    # print('mcts ', time.time() - t)

    fast = FastMeSolver(d)
    fast.solve()

    # t1 = time.time()
    # heuristic_ = HeuristicSearch3(torch.tensor(d).to(torch.float).to("cuda:0"), dgn, width=30, distribution_runs=300)
    # heuristic_.solve()
    # # print(heuristic_.obj_val)
    # t1 = time.time() - t1

    # t = time.time()
    # bb_solver = BB_Solver(d)
    # bb_solver.solve()
    # print(bb_solver.nodes)
    # print('done', time.time() - t)

    # t = time.time()
    # heuristic = HeuristicSearch(torch.tensor(d).to(torch.float).to("cuda:0"), dgn, 4)
    # heuristic.solve()
    # print(heuristic.obj_val)
    # print('')
    # t = time.time() - t
    #
    # pardi = CPPSolver(d[:dim, :dim])
    # pardi.solve()
    # bet = fast.obj_val > mcts_solver.obj_val
    # wor = fast.obj_val < mcts_solver.obj_val
    # outcome = 'better' if bet else ('worse' if wor else 'equal')
    # print(_,  '  swa ', swa.obj_val, '  nj ', nj_phylo.obj_val, '  fast ',  fast.obj_val,
    #       '  mcts ', mcts_solver.obj_val) #, pardi.obj_val, mcts_solver.obj_val == pardi.obj_val, outcome)

    # better.append(bet)
    # worse.append(wor)

print(np.mean(better))
print(np.mean(worse))

# instance = IPSolver(d[:dim, :dim])
# instance.solve(init_adj_sol=swa.solution, logs=True)

