import torch

from Solvers.UCTSolver.node_torch import NodeTorch


def nni_landscape_batch(adj_mat, n_taxa, mat_size):
    batch_size = adj_mat.shape[0]
    mask = torch.triu(torch.ones_like(adj_mat))
    max_taxa = int((mat_size + 2) / 2)
    mask[:, :max_taxa, :] = 0
    internal_branches = torch.nonzero((adj_mat * mask))
    step = len(internal_branches) // batch_size
    internal_branches = torch.hstack([internal_branches, internal_branches[:, torch.tensor([0, 2, 1])]]).view((-1, 3))
    internal_branches[:, 0] = torch.arange(0, len(internal_branches)) // 2

    nni_switches = torch.repeat_interleave(adj_mat, step, dim=0)
    nni_switches[(internal_branches[:, 0], internal_branches[:, 1], internal_branches[:, 2])] = 0

    nni_switches = torch.nonzero(nni_switches[(internal_branches[:, 0], internal_branches[:, 1])])
    nni_switches[:, 0] = torch.repeat_interleave(internal_branches[:, 1], repeats=2)
    nni_switches = nni_switches.reshape((-1, 8))
    to_remove = nni_switches[:, [0, 1, 4, 5, 2, 3, 6, 7]].reshape((-1, 2))
    nni_switches = nni_switches[:, [0, 5, 4, 1, 2, 7, 6, 3]].reshape((-1, 2))
    idxs = torch.arange(len(nni_switches)) // 2

    new_trees = torch.repeat_interleave(adj_mat, 2 * step, dim=0)
    new_trees[(idxs, nni_switches[:, 0], nni_switches[:, 1])] = 1
    new_trees[(idxs, nni_switches[:, 1], nni_switches[:, 0])] = 1
    new_trees[(idxs, to_remove[:, 0], to_remove[:, 1])] = 0
    new_trees[(idxs, to_remove[:, 1], to_remove[:, 0])] = 0

    return new_trees.view((batch_size, 2*step, mat_size, mat_size))


def run_nni_search_batch(iterations, current_adj, best_val, d, n_taxa, m, device):
    sol = current_adj
    batch_size = current_adj.shape[0]
    improved = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)
    for _ in range(iterations):
        expl_trees = nni_landscape_batch(sol, n_taxa, m)
        obj_vals = NodeTorch.compute_obj_val_batch(expl_trees.view(batch_size*expl_trees.shape[1], m , m), d, n_taxa)
        new_obj_val = torch.min(obj_vals.view(batch_size, -1), dim=-1)
        sol = expl_trees[range(batch_size), new_obj_val[1]]
        idxs = torch.argwhere(best_val > new_obj_val.values).squeeze(1)
        current_adj[idxs] =  sol[idxs]
        best_val[idxs] = new_obj_val.values[idxs]
        improved[idxs] = True
    return improved, best_val, current_adj