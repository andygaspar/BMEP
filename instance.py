import networkx as nx
from matplotlib import pyplot as plt
import string
from ip_solver import solve
from Pardi.pardi import Pardi


class Instance:

    def __init__(self, d, labels=None):
        self.d = d
        self.labels = [i for i in string.ascii_uppercase] if labels is None else labels
        self.problem, self.obj_val, self.T = self.solve()
        self.pardi = Pardi(self.T)
        self.graph = self.pardi.get_graph()

    def solve(self):
        problem, T = solve(self.d)
        obj_val = self.problem.getObjective().getValue()
        return problem, obj_val, T

    def print_graph(self):
        nx.draw(self.graph, node_color=[self.graph.nodes[node]["color"] for node in self.graph.nodes],
                with_labels=True, font_weight='bold')
        plt.show()
