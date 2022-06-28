import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from Net.network import Network


class NodeEncoder(nn.Module):
    def __init__(self, din, hidden_dim, device=None):
        super(NodeEncoder, self).__init__()
        self.device = device
        self.fc = nn.Linear(din, hidden_dim).to(self.device)

    def forward(self, x):
        embedding = F.leaky_relu(self.fc(x))
        return embedding


class Message(nn.Module):
    def __init__(self, hidden_dim, dout, device):
        self.device = device
        super(Message, self).__init__()
        self.f_alpha = nn.Linear(hidden_dim * 2, hidden_dim).to(self.device)
        self.v = nn.Linear(hidden_dim, 1).to(self.device)

    def forward(self, hi, hj, mat, mat_mask):
        a = self.f_alpha(torch.cat([hi, hj], dim=-1))
        a = self.v(a).view(mat.shape)
        alpha = F.softmax(torch.mul(a, mat) - 9e15 * (1 - mat_mask), dim=-2)
        return alpha


class FD(nn.Module):
    def __init__(self, hidden_dim, device):
        self.device = device
        super(FD, self).__init__()
        self.fe = nn.Linear(hidden_dim * 3, hidden_dim).to(self.device)
        self.fd = nn.Linear(1, hidden_dim).to(self.device)

    def forward(self, hi, hj, d):
        dd = d.view(d.shape[0], d.shape[1]**2, 1)
        d_ij = self.fd(dd)
        out = F.leaky_relu(self.fe(torch.cat([hi, hj, d_ij], dim=-1)))
        return out


class F_ALL(nn.Module):
    def __init__(self, hidden_dim, device):
        self.device = device
        super(F_ALL, self).__init__()
        self.f = nn.Linear(hidden_dim * 2, hidden_dim).to(self.device)

    def forward(self, hi, hj):
        out = F.leaky_relu(self.f(torch.cat([hi, hj], dim=-1)))
        return out


class FA(nn.Module):
    def __init__(self, hidden_dim, dout, device):
        self.device = device
        super(FA, self).__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim).to(self.device)
        self.fc2 = nn.Linear(hidden_dim, dout).to(self.device)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        q = self.fc2(x)
        return q


class MessageNode(nn.Module):
    def __init__(self, hidden_dim, dout, device):
        self.device = device
        super(MessageNode, self).__init__()
        self.fmn = nn.Linear(hidden_dim * 4, dout).to(self.device)

    def forward(self, h, m1, m2, m3):
        return F.leaky_relu(self.fmn(torch.cat([h, m1, m2, m3], dim=-1)))


class GNN(Network):
    def __init__(self, num_inputs, att_inputs, hidden_dim, num_messages=3, network=None):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask = torch.ones((10, 10)).to(self.device)
        self.agent_din = num_inputs
        self.att_din = att_inputs
        self.hidden_dim = hidden_dim
        self.rounds = num_messages

        self.encoder = NodeEncoder(num_inputs, hidden_dim, self.device)

        self.fd = nn.ModuleList([FD(hidden_dim, self.device) for _ in range(self.rounds)])
        self.ft = nn.ModuleList([F_ALL(hidden_dim, self.device) for _ in range(self.rounds)])
        self.fall = nn.ModuleList([F_ALL(hidden_dim, self.device) for _ in range(self.rounds)])

        self.alpha_d = nn.ModuleList([Message(hidden_dim, hidden_dim, self.device) for _ in range(self.rounds)])
        self.alpha_t = nn.ModuleList([Message(hidden_dim, hidden_dim, self.device) for _ in range(self.rounds)])
        self.alpha_all = nn.ModuleList([Message(hidden_dim, hidden_dim, self.device) for _ in range(self.rounds)])

        # self.gru = torch.nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True).to(self.device)

        self.fmn = MessageNode(hidden_dim, hidden_dim, self.device)
        self.fa = FA(hidden_dim, hidden_dim, self.device)

        if network is not None:
            self.load_weights(network)

    def forward(self, adj_mats, d_mats, initial_masks, masks):
        d_mats = d_mats * 10
        d_mask = d_mats.clone()
        d_mask[d_mask > 0] = 1
        ones = torch.ones_like(adj_mats) # to change when different input size
        h = self.encoder(initial_masks)
        h_0 = h.clone()
        # e_shape = (initial_masks.shape[0], initial_masks.shape[1], initial_masks.shape[1] - 1, self.att_din)
        # e = initial_masks[:, :, self.agent_din:].reshape(e_shape)
        # e = self.edge_encoder(e)

        h = self.message_round(h, d_mats, d_mask, adj_mats, ones, self.rounds)

        h = self.fa(torch.cat([h_0, h], dim=-1))

        y_h = torch.matmul(h, h.permute(0, 2, 1)) * masks - 9e15 * (1 - masks)
        mat_size = y_h.shape
        y_hat = F.softmax(y_h.view(mat_size[0], -1), dim=-1)

        # y_hat = (y_h.view(y_h.shape[0], -1) / torch.sum(y_h, dim=(-2, -1)).unsqueeze(1)).view(mat_size)

        # y_hat = F.softmax(h, dim=-1).unsqueeze(1)
        return y_hat, h

    def message_round(self, h, d, d_mask, adj_mats, ones, rounds):

        for i in range(rounds):
            hi, hj = self.i_j(h, d.shape)
            alpha_d = self.alpha_d[i](hi, hj, d, d_mask).unsqueeze(-1)
            e_d = self.fd[i](hi, hj, d).view(d.shape[0], d.shape[1], d.shape[2], -1)
            m_1 = (alpha_d * e_d).sum(dim=-2)

            alpha_t = self.alpha_t[i](hi, hj, adj_mats, adj_mats).unsqueeze(-1)
            e_t = self.ft[i](hi, hj).view(d.shape[0], d.shape[1], d.shape[2], -1)
            m_2 = (alpha_t * e_t).sum(dim=-2)

            alpha_all = self.alpha_all[i](hi, hj, ones, ones).unsqueeze(-1)
            e_all = self.ft[i](hi, hj).view(d.shape[0], d.shape[1], d.shape[2], -1)
            m_3 = (alpha_all * e_all).sum(dim=-2)

            # h, _ = self.gru(h.view(in_shape), m.view(m_shape))
            # h = h.view(out_shape)
            h = self.fmn(h, m_1, m_2, m_3)

        return h

    @staticmethod
    def i_j(h, e_shape):
        idx = torch.tensor(range(h.shape[1]))
        idxs = torch.cartesian_prod(idx, idx)
        idxs = idxs[[i for i in range(idxs.shape[0])]]
        # hi = h[:, idxs[:, 0]].view((h.shape[0], e_shape[1], e_shape[2], -1))
        # hj = h[:, idxs[:, 1]].view((h.shape[0], e_shape[1], e_shape[2], -1))
        hi = h[:, idxs[:, 0]]
        hj = h[:, idxs[:, 1]]

        return hi, hj