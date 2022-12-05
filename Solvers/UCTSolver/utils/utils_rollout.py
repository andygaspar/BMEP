import random

import torch

from Solvers.UCTSolver.utils.utc_utils import run_nni_search
from Solvers.UCTSolver.utils.utc_utils_batch import run_nni_search_batch


def swa_policy(node, start, adj_mats: torch.tensor, iteration=None):
    batch_size = adj_mats.shape[0]
    obj_vals = None
    for step in range(start, node._n_taxa):
        idxs_list = torch.nonzero(torch.triu(adj_mats), as_tuple=True)
        idxs_list = (torch.tensor(range(idxs_list[0].shape[0])).to(node._device), idxs_list[1], idxs_list[2])
        minor_idxs = torch.tensor([j for j in range(step + 1)]
                                  + [j for j in range(node._n_taxa, node._n_taxa + step - 1)]).to(node._device)

        repetitions = 3 + (step - 3) * 2

        adj_mats = adj_mats.repeat_interleave(repetitions, dim=0)

        sol = node.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=node._n_taxa)
        obj_vals = node.compute_obj_val_batch(adj_mats[:, minor_idxs, :][:, :, minor_idxs],
                                              node._d[:step + 1, :step + 1].repeat(idxs_list[0].shape[0], 1, 1),
                                              step + 1)
        obj_vals = torch.min(obj_vals.reshape(-1, repetitions), dim=-1)
        adj_mats = sol.unsqueeze(0).view(batch_size, repetitions, adj_mats.shape[1],
                                         adj_mats.shape[2])[range(batch_size), obj_vals.indices, :, :]

    return obj_vals.values, adj_mats


def random_policy(node, start, adj_mats, iteration=None):
    batch_size = adj_mats.shape[0]
    for step in range(start, node._n_taxa):
        choices = 3 + (step - 3) * 2
        idxs_list = torch.nonzero(torch.triu(adj_mats)).reshape(batch_size, -1, 3)
        rand_idxs = random.choices(range(choices), k=batch_size)
        idxs_list = idxs_list[[range(batch_size)], rand_idxs, :]
        idxs_list = (idxs_list[:, :, 0], idxs_list[:, :, 1], idxs_list[:, :, 2])
        adj_mats = node.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=node._n_taxa)

    obj_vals = node.compute_obj_val_batch(adj_mats, node._d.repeat(batch_size, 1, 1), node._n_taxa)
    return obj_vals, adj_mats


def mixed_policy(node, start, adj_mats, iteration):
    if iteration < 30:
        return swa_policy(node, start, adj_mats)
    else:
        return random_policy(node, start, adj_mats)


def swa_nni_policy(node, start=3, adj_mats: torch.tensor = None, every=10, after=20, n_iter=5, n_final_iter=20):
    batch_size = adj_mats.shape[0]
    obj_vals = None
    n_internals = node._n_taxa * 2 -2 - node._n_taxa    # to optimise
    for step in range(start, node._n_taxa):
        idxs_list = torch.nonzero(torch.triu(adj_mats), as_tuple=True)
        idxs_list = (torch.tensor(range(idxs_list[0].shape[0])).to(node._device), idxs_list[1], idxs_list[2])
        minor_idxs = torch.tensor([j for j in range(step + 1)]
                                  + [j for j in range(node._n_taxa, node._n_taxa + step - 1)]).to(node._device)

        repetitions = 3 + (step - 3) * 2

        adj_mats = adj_mats.repeat_interleave(repetitions, dim=0)

        sol = node.add_nodes(adj_mats, idxs_list, new_node_idx=step, n=node._n_taxa)
        obj_vals = node.compute_obj_val_batch(adj_mats[:, minor_idxs, :][:, :, minor_idxs],
                                              node._d[:step + 1, :step + 1].repeat(idxs_list[0].shape[0], 1, 1),
                                              step + 1)
        obj_vals = torch.min(obj_vals.reshape(-1, repetitions), dim=-1)
        adj_mats = sol.unsqueeze(0).view(batch_size, repetitions, adj_mats.shape[1],
                                         adj_mats.shape[2])[range(batch_size), obj_vals.indices, :, :]

        # if step > after and (step - 3) % every == 0:
        if step > 6:

            improved, best_val, best_solution = \
                run_nni_search_batch(n_iter, adj_mats, obj_vals.values, node._d, node._n_taxa, node._m, node._device)

            if torch.any(improved):
                print(step)
                adj_mat = adjust_matrices(best_solution[improved], step, n_internals, node._n_taxa)

    # improved, best_val, best_solution = \
    #     run_nni_search(5, adj_mat.squeeze(0), obj_vals[best_idx], self.d, self.n_taxa, self.m,
    #                    self.device)
    #
    # if improved:
    #     adj_mat = self.adjust_matrix(best_solution, self.n_taxa - 1)
    return obj_vals.values, adj_mats


def adjust_matrices(adj_mat, last_inserted_taxa, n_internals, n_taxa):
    adj_mat = adj_mat.unsqueeze(0).repeat(1, 2, 1, 1)

    # reorder matrix according to Pardi
    for step in range(last_inserted_taxa + n_internals, n_taxa, -1):
        print(adj_mat[:, 1, step, last_inserted_taxa])
        a = torch.nonzero(adj_mat[:, 1, step, last_inserted_taxa])
        idxs = torch.argwhere(adj_mat[:, 1, step, last_inserted_taxa]).squeeze(1)
        adj_mat = permute(adj_mat, step, idxs)

        adj_mat[:, 1, :, last_inserted_taxa] = adj_mat[:, 1, last_inserted_taxa, :] = 0
        idxs = torch.nonzero(adj_mat[:, :, 1, step])
        adj_mat[:, 1, idxs[0], idxs[1]] = adj_mat[:, 1, idxs[1], idxs[0]] = 1
        adj_mat[:, 1, :, step] = adj_mat[:, 1, step, :] = 0

        last_inserted_taxa -= 1
    return adj_mat[0].unsqueeze(0)


def permute(adj_mats, step, idx):
    batch_idxs = range(idx.shape[0])
    adj_mats[batch_idxs, :, step, :] += adj_mats[batch_idxs, :, idx, :]
    adj_mats[batch_idxs, :, idx, :] = adj_mats[batch_idxs, :, step, :] - adj_mats[batch_idxs, :, idx, :]
    adj_mats[batch_idxs, :, step, :] -= adj_mats[batch_idxs, :, idx, :]

    adj_mats[batch_idxs, :, :, step] += adj_mats[batch_idxs, :, :, idx]
    adj_mats[batch_idxs, :, :, idx] = adj_mats[batch_idxs, :, :, step] - adj_mats[batch_idxs, :, :, idx]
    adj_mats[batch_idxs, :, :, step] -= adj_mats[batch_idxs, :, :, idx]
    return adj_mats