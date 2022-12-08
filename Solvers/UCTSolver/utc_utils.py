import torch


def nni_landscape(adj_mat, n_taxa, mat_size):
    mask = torch.triu(torch.ones_like(adj_mat))
    max_taxa = int((mat_size + 2) / 2)
    mask[:max_taxa, :] = 0
    internal_branches = torch.nonzero((adj_mat * mask).unsqueeze(0))

    internal_branches = torch.hstack([internal_branches, internal_branches[:, torch.tensor([0, 2, 1])]]).view((-1, 3))
    internal_branches[:, 0] = torch.arange(0, len(internal_branches)) // 2

    nni_switches = torch.repeat_interleave(adj_mat.unsqueeze(0), (n_taxa - 3), dim=0)
    nni_switches[(internal_branches[:, 0], internal_branches[:, 1], internal_branches[:, 2])] = 0

    nni_switches = torch.nonzero(nni_switches[(internal_branches[:, 0], internal_branches[:, 1])])
    nni_switches[:, 0] = torch.repeat_interleave(internal_branches[:, 1], repeats=2)
    nni_switches = nni_switches.reshape((-1, 8))
    nni_switches[:, 3] = nni_switches[:, 1]
    to_remove = nni_switches[:, [0, 1, 4, 5, 2, 3, 6, 7]].reshape((-1, 2))
    nni_switches = nni_switches[:, [0, 5, 4, 1, 2, 7, 6, 3]].reshape((-1, 2))
    idxs = torch.arange(len(nni_switches)) // 2

    new_trees = torch.repeat_interleave(adj_mat.unsqueeze(0), 2 * (n_taxa - 3), dim=0)
    new_trees[(idxs, nni_switches[:, 0], nni_switches[:, 1])] = 1
    new_trees[(idxs, nni_switches[:, 1], nni_switches[:, 0])] = 1
    new_trees[(idxs, to_remove[:, 0], to_remove[:, 1])] = 0
    new_trees[(idxs, to_remove[:, 1], to_remove[:, 0])] = 0

    return new_trees
