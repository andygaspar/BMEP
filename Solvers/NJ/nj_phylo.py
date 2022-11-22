import networkx as nx
import numpy as np
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

from Solvers.solver import Solver


class NjPhylo(Solver):

    def __int__(self, d):
        super().__init__(d)

    def solve(self):

        d = [el[:i + 1] for i, el in enumerate(self.d.tolist())]
        dm = DistanceMatrix([str(i) for i in range(self.n_taxa)], d)
        constructor = DistanceTreeConstructor()
        tree = constructor.nj(dm)
        t = Phylo.to_networkx(tree)
        sol = nx.adjacency_matrix(Phylo.to_networkx(tree)).toarray()
        sol[sol != 0] = 1
        names = []
        for node in t.nodes():
            names.append(node.name)

        order = np.argsort(names)
        self.solution = np.zeros_like(sol)
        for i in order:
            for j in order:
                self.solution[i, j] = sol[order[i], order[j]]

        self.obj_val = self.compute_obj_val_from_adj_mat(self.solution, self.d, self.n_taxa)
