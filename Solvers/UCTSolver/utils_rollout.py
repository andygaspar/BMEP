import random

import torch


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
    if iteration < 10:
        return swa_policy(node, start, adj_mats)
    elif iteration % 20 != 0:
        return random_policy(node, start, adj_mats)
    else:
        return swa_policy(node, start, adj_mats)
