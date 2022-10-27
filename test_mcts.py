import time

from Data_.Datasets.bmep_dataset import BMEP_Dataset

from Solvers.UCTSolver.utc_solver import UtcSolver
from Solvers.PardiSolver.pardi_solver import PardiSolver
from Solvers.SWA.swa_solver import SwaSolver
from Solvers.solver import Solver

funs = Solver()
add_node = funs.add_node
compute_obj_val = funs.compute_obj_val_from_adj_mat

folder = 'GNN_TAU'
file = '_3.622'
data_folder = '6_taxa_0'




data_ = BMEP_Dataset(folder_name=data_folder)

d = data_.d_mats[0].to("cpu").numpy()

t = time.time()
mcts_solver = UtcSolver(d)
mcts_solver.solve(100)
print('mcts ', time.time() - t)

t = time.time()
swa = SwaSolver(d)
swa.solve()
print('swa ', time.time() - t)

t = time.time()
pardi = PardiSolver(d[:6, :6])
pardi.solve()
print('pardi ', time.time() - t)

print(mcts_solver.obj_val)
print(swa.obj_val)
print(pardi.obj_val)


