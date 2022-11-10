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
        # x = nn.functional.dropout(x, p=self.drop_out)
        q = self.fc2(x)
        return q


class EGAT_MH_RL(Network):
    def __init__(self, net_params, network=None, normalised=False):
        super().__init__(net_params["normalisation factor"])
        self.taxa_inputs, self.internal_inputs, self.edge_inputs = \
            net_params["taxa_inputs"], net_params["internal_inputs"], net_params["edge_inputs"]
        self.embedding_dim, self.hidden_dim = net_params["embedding_dim"], net_params["hidden_dim"]
        self.rounds, self.num_heads = net_params["num_messages"], net_params["num_heads"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.normalised = normalised


        self.taxa_encoder = Encoder(self.taxa_inputs, self.embedding_dim, self.hidden_dim, self.device)
        self.internal_encoder = Encoder(self.internal_inputs, self.embedding_dim, self.hidden_dim, self.device)
        self.edge_encoder = Encoder(self.edge_inputs, self.embedding_dim, self.hidden_dim, self.device)

        attention_dims = [self.embedding_dim]
        for i in range(self.rounds):
            attention_dims.append(attention_dims[-1]*self.num_heads)

        self.W = nn.ModuleList([nn.Linear(attention_dims[i], attention_dims[i + 1], bias=False).to(self.device)
                                for i in range(self.rounds)])
        self.a = [nn.Parameter(torch.Tensor(1, 1, attention_dims[i + 1] * 3)).to(self.device)
                         for i in range(self.rounds)]

        # self.drop_out = net_params['drop out']
        self.drop_out = None

        self.fa = FA(attention_dims[-1], attention_dims[-1], self.drop_out, self.device)

        if network is not None:
            self.load_weights(network)

    def forward(self, data):
        taxa, internal, messages, masks = data
        if not self.normalised:
            taxa = taxa / self.normalisation_factor
            internal = internal / self.normalisation_factor
            messages = messages / self.normalisation_factor

        a = self.taxa_encoder(taxa)
        b = self.internal_encoder(internal)

        h = torch.cat([a, b], dim=1)
        message = self.edge_encoder(messages)

        for i in range(self.rounds):
            z = self.W[i](h)
            e_i = z.repeat_interleave(z.shape[1], 1)
            e_j = z.repeat(1, z.shape[1], 1)
            e = nn.functional.leaky_relu((self.a[i] * torch.cat([e_i, e_j, message], dim=-1)).sum(dim=-1))
            alpha = nn.functional.softmax(e.view(-1, z.shape[1], z.shape[1]), dim=-1)
            h = torch.tanh(torch.matmul(alpha, z))


        y_h = torch.matmul(h, h.permute(0, 2, 1)) * masks - 9e15 * (1 - masks)
        mat_size = y_h.shape
        y_hat = F.softmax(y_h.view(mat_size[0], -1), dim=-1)

        return y_hat, F.log_softmax(y_h.view(mat_size[0], -1), dim=-1)



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
    taxa_input = np.zeros((m, 7))
    d_means = mat.mean(axis=1)
    taxa_input[:n_taxa, 0] = d_means
    taxa_input[:n_taxa, 1] = mat.std(axis=1)
    taxa_input[:n_taxa, 2] = mat.sum(axis=1)/step
    taxa_input[:n_taxa, 3] = np.sqrt(((mat - mat.mean(axis=1))**2).sum(axis=1))/step

    for i in range(step):
        taxa_input[i, 4] = get_taxa_standing(d, tau, i, n_taxa)

    for j in range(n_taxa, n_taxa + step - 3):
        taxa_input[j, 5] = get_internal_standing_d(d_means, tau, j)
        taxa_input[j, 6] = get_internal_standing(tau, j)
    return torch.tensor(taxa_input).to(torch.float).to("cuda:0")

taxa = make_node_input(mat, n_taxa, tau, step).unsqueeze(0)
internals = torch.ones((4, 2)).to(torch.float).to("cuda:0").unsqueeze(0)
messages = torch.ones((16**2, 2)).to(torch.float).to("cuda:0").unsqueeze(0)

# for i in range(5):
#     print(get_taxa_standing(mat[:5, :5], tau, i))


net_params = {'normalisation factor': np.max(mat), 'embedding_inputs': 6, 'edge_inputs': 2,  'embedding_dim': 4,
              'hidden_dim': 6, 'num_messages': 3, 'num_heads': 2, 'taxa_inputs': 7, 'internal_inputs': 2}

net = EGAT_MH_RL(net_params)
net((taxa, internals, messages, None))

