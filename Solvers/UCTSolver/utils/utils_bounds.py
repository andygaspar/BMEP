import torch


def compute_greedy_bound(node):
    minor_idxs = torch.tensor([j for j in range(node._step_i)]
                              + [j for j in range(node._n_taxa, node._n_taxa + node._step_i - 2)]).to(node._device)
    adj_mat = node._adj_mat[:, minor_idxs, :][:, :, minor_idxs].squeeze(0)
    Tau = torch.full_like(adj_mat, node._step_i)
    Tau[adj_mat > 0] = 1
    diag = torch.eye(adj_mat.shape[1], dtype=torch.bool)
    Tau[diag] = 0  # diagonal elements should be zero
    for i in range(adj_mat.shape[1]):
        # The second term has the same shape as Tau due to broadcasting
        Tau = torch.minimum(Tau, Tau[i, :].unsqueeze(0).repeat(adj_mat.shape[1], 1)
                            + Tau[:, i].unsqueeze(1).repeat(1, adj_mat.shape[1]))
    d = node._d[: node._step_i, :node._step_i]
    first_term = (d * 2 ** (-(Tau[:node._step_i, :node._step_i] + node._n_taxa - node._step_i))).sum()
    tau_max_distance = 2 ** (-(torch.max(Tau) + node._n_taxa - node._step_i))
    diag = torch.eye(node._d.shape[1])
    min_lower_d = torch.min((node._d + diag)[node._step_i:, :], dim=-1)[0]
    min_up_right_d = torch.min((node._d + diag)[: node._step_i:, node._step_i:], dim=-1)[0]

    return first_term + (min_up_right_d.sum() + min_lower_d.sum()) * tau_max_distance