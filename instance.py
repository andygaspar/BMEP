import networkx as nx
from matplotlib import pyplot as plt
import string
from ip_solver import solve
from print_graph import get_graph


class Instance:

    def __init__(self, d, labels=None):
        self.d = d
        self.labels = [i for i in string.ascii_uppercase] if labels is None else labels
        self.problem = None
        self.obj_val = None
        self.T = None
        self.graph = None
        self.solve()
        self.get_graph()

    def solve(self):
        self.problem, self. T = solve(self.d)
        self.obj_val = self.problem.getObjective().getValue()

    def get_graph(self):
        self.graph = get_graph(self.T, self.labels)

    def print_graph(self):
        nx.draw(self.graph, node_color=[self.graph.nodes[node]["color"] for node in self.graph.nodes],
                with_labels=True, font_weight='bold')
        plt.show()
