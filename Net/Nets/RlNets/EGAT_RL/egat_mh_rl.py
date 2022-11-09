import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from Net.network import Network


class Encoder(nn.Module):
    def __init__(self, din, h_dimension, hidden_dim, device=None):
        super(Encoder, self).__init__()
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(din, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, h_dimension),
            nn.Tanh()
        ).to(self.device)

    def forward(self, x):
        return self.fc(x)

class FA(nn.Module):
    def __init__(self, h_dimension, hidden_dim, drop_out, device):
        self.device = device
        super(FA, self).__init__()
        self.fc1 = nn.Linear(h_dimension, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(hidden_dim, h_dimension).to(self.device)
        self.drop_out = drop_out

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = nn.functional.dropout(x, p=self.drop_out)
        q = self.fc2(x)
        return q


class EGAT_MH_RL(Network):
    def __init__(self, net_params, network=None):
        super().__init__(net_params["normalisation factor"])
        self.node_inputs, self.edge_inputs = net_params["node_inputs"], net_params["edge_inputs"]
        self.h_dimension, self.hidden_dim = net_params["h_dimension"], net_params["hidden_dim"]
        self.rounds, self.num_heads = net_params["num_messages"], net_params["num_heads"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.mask = torch.ones((10, 10)).to(self.device)

        self.node_encoder = Encoder(self.node_inputs, self.h_dimension, self.hidden_dim, self.device)
        self.edge_inputs = Encoder(self.edge_inputs, self.h_dimension, self.hidden_dim, self.device)

        self.W = nn.ModuleList([nn.Linear(self.h_dimension, self.num_heads * self.h_dimension, bias=False)
                                for _ in range(self.rounds)])
        self.a_target = [nn.Parameter(torch.Tensor(1, self.num_heads, self.hidden_dim)) for _ in range(self.rounds)]

        self.drop_out = net_params['drop out']

        self.fa = FA(self.h_dimension, self.hidden_dim, self.drop_out, self.device)

        if network is not None:
            self.load_weights(network)

    def forward(self, data):
        adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, taus, tau_masks, y = data
        d_mats_ = d_mats / self.normalisation_factor

        h = self.encoder(initial_masks)
        taus[taus > 0] = 1 / taus[taus > 0]
        h = self.context_message(h, d_mats_, d_masks, initial_masks, 3)
        h = self.tau_message(h, taus, tau_masks, ad_masks, 3)
        h = self.fa(h)

        y_h = torch.matmul(h, h.permute(0, 2, 1)) * masks - 9e15 * (1 - masks)
        mat_size = y_h.shape
        y_hat = F.softmax(y_h.view(mat_size[0], -1), dim=-1)

        return y_hat, F.log_softmax(y_h.view(mat_size[0], -1), dim=-1)

    def context_message(self, h, d, d_mask, initial_mask, rounds):
        for i in range(rounds):
            hi, hj = self.i_j(h)
            alpha_d = self.alpha_d[i](hi, hj, d, d_mask).unsqueeze(-1)
            e_d = self.fd[i](hi, hj, d).view(d.shape[0], d.shape[1], d.shape[2], -1)
            m_1 = (alpha_d * e_d).sum(dim=-2)
            hd = initial_mask[:, :, 0].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h_not_d = initial_mask[:, :, 1].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h = self.fm1(h, m_1) * hd + h * h_not_d

        return h

    def tau_message(self, h, taus, tau_masks, ad_masks, rounds):
        for i in range(rounds):
            hi, hj = self.i_j(h)
            alpha_t = self.alpha_t[i](hi, hj, taus, tau_masks).unsqueeze(-1)
            e_d = self.ft[i](hi, hj, taus).view(taus.shape[0], taus.shape[1], taus.shape[2], -1)
            m_2 = (alpha_t * e_d).sum(dim=-2)
            h_adj = ad_masks[:, :, 0].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h_not_adj = ad_masks[:, :, 1].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h = self.fm2(h, m_2) * h_adj + h * h_not_adj
        return h

    @staticmethod
    def i_j(h):
        idx = torch.tensor(range(h.shape[1]))
        idxs = torch.cartesian_prod(idx, idx)
        idxs = idxs[[i for i in range(idxs.shape[0])]]
        # hi = h[:, idxs[:, 0]].view((h.shape[0], e_shape[1], e_shape[2], -1))
        # hj = h[:, idxs[:, 1]].view((h.shape[0], e_shape[1], e_shape[2], -1))
        hi = h[:, idxs[:, 0]]
        hj = h[:, idxs[:, 1]]

        return hi, hj


import os
import numpy as np
from Solvers.SWA.swa_solver import SwaSolver

def get_taxa_standing(d, tau, taxa_idx, n_taxa):
    return sum(d[taxa_idx] / 2 ** (tau[taxa_idx, n_taxa]))

def get_internal_standing_d(d_means, tau_full, internal_idx):
    return np.mean(d_means / 2 ** tau_full[internal_idx, :d_means.shape[0]])

def get_internal_standing(tau_full, internal_idx):
    return np.mean(tau_full[internal_idx] / 2 ** tau_full[internal_idx])




path = 'Data_/csv_'
filenames = sorted(next(os.walk(path), (None, None, []))[2])

mats = []
for file in filenames:
    if file[-4:] == '.txt':
        mats.append(np.genfromtxt('Data_/csv_/' + file, delimiter=','))

n_taxa = 7
step = 2

mat = mats[2][:n_taxa, :n_taxa]/np.max(mats[2][:n_taxa, :n_taxa])

swa = SwaSolver(mat[:step + 3, :step + 3])
swa.solve()
tau_partial = swa.get_tau(swa.solution)
tau = np.zeros((n_taxa*2 -2, n_taxa*2-2))
tau[:step+3, :step+3] = tau_partial[:step+3, :step+3]
tau[n_taxa: n_taxa+3, :n_taxa] = tau_partial[n_taxa-step:, :n_taxa]

tau[:n_taxa, n_taxa: n_taxa+3] = tau_partial[:n_taxa, n_taxa-step:]
y = tau_partial[n_taxa-step:, n_taxa-step:]
tau[n_taxa: n_taxa+3, n_taxa: n_taxa+3] = tau_partial[n_taxa-step:, n_taxa-step:]

def make_node_input(d, n_taxa, tau, step):
    step += 3
    m = n_taxa*2 - 2
    node_input = np.zeros((m, 7))
    d_means = mat.mean(axis=1)
    node_input[:n_taxa, 0] = d_means
    node_input[:n_taxa, 1] = mat.std(axis=1)
    node_input[:n_taxa, 2] = mat.sum(axis=1)/step
    node_input[:n_taxa, 3] = np.sqrt(((mat - mat.mean(axis=1))**2).sum(axis=1))/step

    for i in range(step):
        node_input[i, 4] = get_taxa_standing(d, tau, i, n_taxa)

    for j in range(n_taxa, n_taxa + step - 3):
        node_input[j, 5] = get_internal_standing_d(d_means, tau, j)
        node_input[j, 6] = get_internal_standing(tau, j)
    print(node_input)

make_node_input(mat, n_taxa, tau, step)

# for i in range(5):
#     print(get_taxa_standing(mat[:5, :5], tau, i))


net_params = {'normalisation factor': np.max(mat), 'node_inputs': 6, 'edge_inputs': 5,  'h_dimension': 128, 'hidden_dim': 64, 'num_messages': 3, 'num_heads': 3}


