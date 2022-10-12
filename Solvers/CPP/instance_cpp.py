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

    def solve(self, d, n, log):
        self.lib.run.restype = ndpointer(dtype=ctypes.c_int32, shape=(n*2 - 2, n*2 - 2))
        return self.lib.run(d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n, log)




class CppSolver(Solver):

    def __init__(self, d):
        super().__init__(d)

    def solve(self, log):
        cpp = InstanceCpp()
        # print(self.d)
        d = np.ascontiguousarray(self.d.flatten(), dtype=np.double)
        # print("a")
        adj_mat = cpp.solve(d, self.m, log)
        # print("b")
        g = nx.from_numpy_matrix(adj_mat)
        self.T = nx.floyd_warshall_numpy(g)[:self.m, :self.m]
        return self.compute_obj(), self.T
