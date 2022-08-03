import string
import numpy as np
import networkx as nx
from Pardi.leaf import Leaf


class GraphAndData:

    def __init__(self):
        self.graph = nx.Graph()

    def get_graph(self, leaves, n):
        self.graph = nx.Graph()
        self.graph.add_nodes_from([(leaf.label, {"color": "red"}) for leaf in leaves])
        self.graph.add_nodes_from([(k, {"color": "green"}) for k in range(1, n - 1)])
        self.graph.add_edges_from([(leaves[0].label, 1), (leaves[1].label, 1), (leaves[2].label, 1)])
        adj_mats = [nx.adjacency_matrix(self.graph).toarray()]

        for k, leaf in enumerate(leaves[3:]):
            assignment = leaf.get_assignment()
            if not type(assignment) == Leaf:
                internal_node = k + 2
                self.graph.remove_edge(assignment[0], assignment[1])
                self.graph.add_edges_from([(leaf.label, internal_node), (assignment[0], internal_node),
                                           (assignment[1], internal_node)])

            else:
                internal_node = k + 2
                removing_edge = list(self.graph.edges(assignment.label))[0]
                self.graph.remove_edge(removing_edge[0], removing_edge[1])
                self.graph.add_edges_from([(leaf.label, internal_node), (assignment.label, internal_node),
                                           (internal_node, removing_edge[1])])
            adj_mats.append(nx.adjacency_matrix(self.graph).toarray())
        return self.graph, adj_mats
