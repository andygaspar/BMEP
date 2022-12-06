import networkx as nx
import matplotlib.pyplot as plt
from Bio import Phylo

from Solvers.FastME.pharser_newik.newwik_handler import get_adj_from_nwk


def recursive_search(dict, key):
    if key in dict:
        return dict[key]
    for k, v in dict.items():
        item = recursive_search(v, key)
        if item is not None:
            return item

def bfs_edge_lst(graph, n):
    return list(nx.bfs_edges(graph, n))

def load_graph(filename):
    G = nx.Graph()
    # build the graph
    return G

def tree_from_edge_lst(elst):
    tree = {1: {}}
    for src, dst in elst:
        subt = recursive_search(tree, src)
        subt[dst] = {}
    return tree

def tree_to_newick(tree):
    items = []
    for k in tree.keys():
        s = ''
        if len(tree[k].keys()) > 0:
            subt = tree_to_newick(tree[k])
            if subt != '':
                s += '(' + subt + ')'
        s += str(k)
        items.append(s)
    return ','.join(items)

tree = Phylo.read('Solvers/FastME/fastme-2.1.6.4/mat.mat_fastme_tree.nwk', "newick")
adj = get_adj_from_nwk('Solvers/FastME/fastme-2.1.6.4/mat.mat_fastme_tree.nwk')
g = nx.from_numpy_matrix(adj)
elst = bfs_edge_lst(g, 1)
tree = tree_from_edge_lst(elst)
newick = tree_to_newick(tree) + ';'
print(newick)