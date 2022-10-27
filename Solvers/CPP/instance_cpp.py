import contextlib
import copy
import ctypes
import io

import networkx as nx
import numpy as np
import xpress
from numpy.ctypeslib import ndpointer

from Solvers.solver import Solver


class InstanceCpp:

    def __init__(self):
        self.lib = ctypes.CDLL('Solvers/CPP/bridge.so')
        self.lib.run.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_bool]
        self.lib.run.restype = ctypes.c_void_p

    def solve(self, d, n_taxa, log):
        self.lib.run.restype = ndpointer(dtype=ctypes.c_int32, shape=(n_taxa*2 - 2, n_taxa*2 - 2))
        return self.lib.run(d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n_taxa, log)


cpp = InstanceCpp()


class CppSolver(Solver):

    def __init__(self, d):
        super().__init__(d)

    def solve(self, log):
        d = np.ascontiguousarray(self.d.flatten(), dtype=np.float)
        adj_mat = cpp.solve(d, self.n_taxa, log)
        g = nx.from_numpy_matrix(adj_mat)
        self.T = nx.floyd_warshall_numpy(g)[:self.n_taxa, :self.n_taxa]
        return self.compute_obj(), self.T
