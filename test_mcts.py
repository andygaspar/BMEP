import json
import time

import networkx as nx
import numpy as np

from Data_.Datasets.bmep_dataset import BMEP_Dataset

from torch.utils.data import DataLoader

from Net.Nets.GNN1.gnn_1 import GNN_1
from Net.network_manager import NetworkManager
from Solvers.MCTS.mcts import MctsSolver, State
from Solvers.NetSolvers.heuristic_search import HeuristicSearch
from Solvers.NetSolvers.heuristic_search_2 import HeuristicSearch2
from Solvers.NetSolvers.net_solver import NetSolver
from Solvers.NetSolvers.search_solver import SearchSolver
from Solvers.PardiSolver.pardi_solver import PardiSolver
from Solvers.SWA.swa_solver import SwaSolver
from Solvers.solver import Solver

funs = Solver()
add_node = funs.add_node
compute_obj_val = funs.compute_obj_val_from_adj_mat

folder = 'GNN_TAU'
file = '_3.622'
data_folder = '6_taxa_0'


def swa_policy(state: State):
    swa_ = SwaSolver(state.d)
    swa_.solve(start=state.step, adj_mat=state.adj_mat)
    return swa_.obj_val


data_ = BMEP_Dataset(folder_name=data_folder)

d = data_.d_mats[0].to("cpu").numpy()
mcts_solver = MctsSolver(d)
mcts_solver.solve(100, rolloutPolicy=swa_policy)





swa = SwaSolver(d)
swa.solve()

pardi = PardiSolver(d[:6, :6])
pardi.solve()

print(mcts_solver.obj_val)
print(swa.obj_val)
print(pardi.obj_val)



