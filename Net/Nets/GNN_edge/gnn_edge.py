import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from Net.network import Network


class NodeEncoder(nn.Module):
    def __init__(self, din, h_dimension, device=None):
        super(NodeEncoder, self).__init__()
        self.device = device
        self.fc = nn.Linear(din, h_dimension).to(self.device)

    def forward(self, x):
        embedding = torch.tanh(self.fc(x))
        return embedding


class Message(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        self.device = device
        super(Message, self).__init__()
        self.f_alpha = nn.Linear(h_dimension * 2, hidden_dim).to(self.device)
        self.v = nn.Linear(hidden_dim, 1).to(self.device)

    def forward(self, hi, hj, mat, mat_mask):
        a = torch.tanh(self.f_alpha(torch.cat([hi, hj], dim=-1))) #messo lrelu
        a = torch.tanh(self.v(a).view(mat.shape)) #messo lrelu
        alpha = F.softmax(torch.mul(a, mat) - 9e15 * (1 - mat_mask), dim=-1)
        return alpha


class FD(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        self.device = device
        super(FD, self).__init__()
        self.fe = nn.Linear(h_dimension * 2 + hidden_dim, hidden_dim).to(self.device)
        self.fd = nn.Linear(1, hidden_dim).to(self.device)

    def forward(self, hi, hj, d):
        dd = d.view(d.shape[0], d.shape[1] ** 2, 1)
        d_ij = torch.tanh(self.fd(dd)) #messo lrelu
        out = torch.tanh(self.fe(torch.cat([hi, hj, d_ij], dim=-1)))
        return out


class F_ALL(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        self.device = device
        super(F_ALL, self).__init__()
        self.f = nn.Linear(h_dimension * 2, hidden_dim).to(self.device)

    def forward(self, hi, hj):
        out = torch.tanh(self.f(torch.cat([hi, hj], dim=-1)))
        return out


class MessageNode(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        self.device = device
        super(MessageNode, self).__init__()
        self.fmn = nn.Linear(h_dimension + hidden_dim, h_dimension).to(self.device)

    def forward(self, h, m1):
        return torch.tanh(self.fmn(torch.cat([h, m1], dim=-1)))


class FA(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        self.device = device
        super(FA, self).__init__()
        self.fc1 = nn.Linear(h_dimension, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(hidden_dim, h_dimension).to(self.device)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        q = self.fc2(x)
        return q


class Edge1(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        self.device = device
        super(Edge1, self).__init__()
        self.fc1 = nn.Linear(h_dimension * 2, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(hidden_dim, h_dimension).to(self.device)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        q = torch.tanh(self.fc2(x))
        return q


class Edge2(nn.Module):
    def __init__(self, h_dimension, hidden_dim, device):
        self.device = device
        super(Edge2, self).__init__()
        self.fc1 = nn.Linear(h_dimension, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(hidden_dim, 1).to(self.device)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        q = torch.tanh(self.fc2(x))
        return q


class GNN_edge(Network):
    def __init__(self, net_params, network=None):
        super().__init__()
        num_inputs, h_dimension, hidden_dim, num_messages = net_params["num_inputs"], net_params["h_dimension"], \
                                                            net_params["hidden_dim"], net_params["num_messages"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask = torch.ones((10, 10)).to(self.device)

        self.rounds = num_messages

        self.encoder = NodeEncoder(num_inputs, h_dimension, self.device)

        self.fd = nn.ModuleList([FD(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])
        self.ft = nn.ModuleList([F_ALL(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])
        self.fall = nn.ModuleList([F_ALL(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])

        self.alpha_d = nn.ModuleList([Message(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])
        self.alpha_t = nn.ModuleList([Message(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])
        self.alpha_all = nn.ModuleList([Message(h_dimension, hidden_dim, self.device) for _ in range(self.rounds)])

        # self.gru = torch.nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True).to(self.device)
        self.f_edge_1 = Edge1(h_dimension, hidden_dim, self.device)
        self.f_edge_2 = Edge2(h_dimension, hidden_dim, self.device)


        self.fm1 = MessageNode(h_dimension, hidden_dim, self.device)
        self.fm2 = MessageNode(h_dimension, hidden_dim, self.device)
        self.fm3 = MessageNode(h_dimension, hidden_dim, self.device)
        self.fa = FA(h_dimension, hidden_dim, self.device)

        if network is not None:
            self.load_weights(network)

    def forward(self, adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks):
        d_mats = d_mats
        h = self.encoder(initial_masks)

        h = self.message_round(h, d_mats, d_masks, adj_mats, size_masks, initial_masks, ad_masks, self.rounds)

        h = self.fa(h)

        hi, hj = self.i_j(h)

        v1 = self.f_edge_1(torch.cat([hi, hj], dim=-1))
        v2 = self.f_edge_1(torch.cat([hj, hi], dim=-1))
        v = (v1 + v2) / 2
        mat_size = int(np.sqrt(v.shape[1]))
        v = self.f_edge_2(v).view(v.shape[0], mat_size, mat_size)

        y_edge = v * masks - 9e15 * (1 - masks)
        y_hat = F.softmax(y_edge.view(v.shape[0], -1), dim=-1)

        return y_hat, v

    def message_round(self, h, d, d_mask, adj_mats, size_masks, initial_mask, ad_masks, rounds):

        for i in range(rounds):
            hi, hj = self.i_j(h)
            alpha_d = self.alpha_d[i](hi, hj, d, d_mask).unsqueeze(-1)
            e_d = self.fd[i](hi, hj, d).view(d.shape[0], d.shape[1], d.shape[2], -1)
            m_1 = (alpha_d * e_d).sum(dim=-2)
            hd = initial_mask[:, :, 0].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h_not_d = initial_mask[:, :, 1].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h = self.fm1(h, m_1) * hd + h * h_not_d

            hi, hj = self.i_j(h)
            alpha_t = self.alpha_t[i](hi, hj, adj_mats, adj_mats).unsqueeze(-1)
            e_t = self.ft[i](hi, hj).view(d.shape[0], d.shape[1], d.shape[2], -1)
            m_2 = (alpha_t * e_t).sum(dim=-2)
            h_adj = ad_masks[:, :, 0].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h_not_adj = ad_masks[:, :, 1].unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h = self.fm2(h, m_2) * h_adj + h * h_not_adj

            hi, hj = self.i_j(h)
            alpha_all = self.alpha_all[i](hi, hj, size_masks, size_masks).unsqueeze(-1)
            e_all = self.ft[i](hi, hj).view(d.shape[0], d.shape[1], d.shape[2], -1)
            m_3 = (alpha_all * e_all).sum(dim=-2)

            # h, _ = self.gru(h.view(in_shape), m.view(m_shape))
            # h = h.view(out_shape)
            h = self.fm3(h, m_3)

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
