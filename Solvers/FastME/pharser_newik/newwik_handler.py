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


def update_distance(T, a, b, n_taxa):
    T[a, :] -= 1
    T[:, a] -= 1
    T[b, :] = T[:, b] = n_taxa + 1
    return T


def compute_newick(T):
    n_taxa = T.shape[0]
    taxa = range(n_taxa)
    taxa_dict = dict(zip(taxa, [str(t) for t in taxa]))
    T = np.copy(T)
    over, newick = False, None
    while not over:
        cherries = np.asarray(T==2).nonzero()
        if cherries[0].shape[0] == np.unique(cherries[0]).shape[0]:
            a, b = cherries[0][0], cherries[1][0]
            new_term = '(' + taxa_dict[a] + ',' + taxa_dict[b] + ')'
            taxa_dict[a] = new_term
            T = update_distance(T, a, b, n_taxa)
        else:
            a, b, c = cherries[0][0], cherries[1][0], cherries[1][1]
            over = True
            newick = '(' + taxa_dict[a] + ',' + taxa_dict[b] + ',' + taxa_dict[c] + ');'

    return newick



