import networkx as nx
import numpy as np
from Bio import Phylo


def permute_np(adj_mats, step, idx):
    adj_mats[:, step, :] += adj_mats[:, idx, :]
    adj_mats[:, idx, :] = adj_mats[:, step, :] - adj_mats[:, idx, :]
    adj_mats[:, step, :] -= adj_mats[:, idx, :]

    adj_mats[:, :, step] += adj_mats[:, :, idx]
    adj_mats[:, :, idx] = adj_mats[:, :, step] - adj_mats[:, :, idx]
    adj_mats[:, :, step] -= adj_mats[:, :, idx]
    return adj_mats


def get_adj_from_nwk(file):
    tree = Phylo.read(file, "newick")
    g = nx.Graph(Phylo.to_networkx(tree))
    m = len(g)
    n_taxa = (m + 2)//2
    last_internal = len(g) - 1
    for node in g.nodes.keys():
        if node.name is None:
            node.name = last_internal
            last_internal -= 1
        else:
            node.name = int(node.name)

    H = nx.Graph()
    H.add_nodes_from(sorted(g.nodes, key=lambda n: n.name))
    H.add_edges_from(g.edges(data=True))
    adj_mat = np.zeros((len(g), len(g)), dtype=int)
    for edge in g.edges:
        a, b = edge
        adj_mat[a.name, b.name] = adj_mat[b.name, a.name] = 1

    adj_mat.sum(axis=1)

    adj_mats = adj_mat[np.newaxis, :, :].repeat(2, axis=0)
    last_inserted_taxa = n_taxa - 1

    # reorder matrix according to Pardi
    for step in range(m - 1, n_taxa, -1):
        if adj_mats[1][step, last_inserted_taxa] == 0:
            idx = np.nonzero(adj_mats[1][last_inserted_taxa])
            idx = idx[0][0]
            adj_mats = permute_np(adj_mats, step, idx)

        adj_mats[1][:, last_inserted_taxa] = adj_mats[1][last_inserted_taxa, :] = 0
        idxs = tuple(np.nonzero(adj_mats[1][step])[0])
        adj_mats[1][idxs[0], idxs[1]] = adj_mats[1][idxs[1], idxs[0]] = 1
        adj_mats[1][:, step] = adj_mats[1][step, :] = 0

        last_inserted_taxa -= 1

    return adj_mats[0]
