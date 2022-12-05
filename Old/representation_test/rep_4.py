import torch
import torch.nn as nn
import torch.nn.functional as F

from ML.Net.network import Network


# USE_CUDA = torch.cuda.is_available()
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
#
#                                                                                                                 **kwargs)


class NodeEncoder(nn.Module):
    def __init__(self, din, hidden_dim, device=None):
        super(NodeEncoder, self).__init__()
        self.device = device
        self.fc = nn.Linear(din, hidden_dim).to(self.device)

    def forward(self, x):
        embedding = F.leaky_relu(self.fc(x))
        return embedding


class MessageEncoder(nn.Module):
    def __init__(self, hidden_dim, hidden_dim_2, device=None):
        super(MessageEncoder, self).__init__()
        self.device = device
        self.fc = nn.Linear(hidden_dim, hidden_dim_2).to(self.device)

    def forward(self, x):
        embedding = F.leaky_relu(self.fc(x))
        return embedding


class ALPHA(nn.Module):
    def __init__(self, hidden_dim, dout, device):
        self.device = device
        super(ALPHA, self).__init__()
        self.f_alpha = nn.Linear(hidden_dim * 2, hidden_dim).to(self.device)
        self.v = nn.Linear(hidden_dim, 1).to(self.device)

    def forward(self, hi, hj, d):
        a = self.f_alpha(torch.cat([hi, hj], dim=-1))
        a = self.v(a).view(d.shape)
        alpha = F.softmax(torch.mul(a, d), dim=-2)
        return alpha


class FE(nn.Module):
    def __init__(self, hidden_dim, dout, device):
        self.device = device
        super(FE, self).__init__()
        self.fe = nn.Linear(hidden_dim * 3, hidden_dim).to(self.device)
        self.fd = nn.Linear(1, hidden_dim).to(self.device)

    def forward(self, hi, hj, d):
        dd = d.view(d.shape[0], d.shape[1]**2, 1)
        d_ij = self.fd(dd)
        out = F.leaky_relu(self.fe(torch.cat([hi, hj, d_ij], dim=-1)))
        return out


class MessageNode(nn.Module):
    def __init__(self, hidden_dim, dout, device):
        self.device = device
        super(MessageNode, self).__init__()
        self.fmn = nn.Linear(hidden_dim * 2, dout).to(self.device)

    def forward(self, h, m):
        return F.leaky_relu(self.fmn(torch.cat([h, m], dim=-1)))


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


class DGN_(Network):
    def __init__(self, num_inputs, att_inputs, hidden_dim, network=None):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask = torch.ones((10, 10)).to(self.device)
        self.agent_din = num_inputs
        self.att_din = att_inputs
        self.hidden_dim = hidden_dim
        self.rounds = 3

        self.encoder = NodeEncoder(num_inputs, hidden_dim, self.device)
        self.edge_encoder = MessageEncoder(att_inputs, hidden_dim, self.device)
        self.fe = nn.ModuleList([FE(hidden_dim, hidden_dim, self.device) for _ in range(self.rounds)])
        self.alpha = nn.ModuleList([ALPHA(hidden_dim, hidden_dim, self.device) for _ in range(self.rounds)])
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True).to(self.device)
        self.fa = FA(hidden_dim, hidden_dim, self.device)
        self.fmn = MessageNode(hidden_dim, hidden_dim, self.device)

        self.final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, 1)
        ).to(self.device)

        if network is not None:
            self.load_weights(network)

    def forward(self, adj_mats, d_mats, initial_masks, masks):
        d_mats = d_mats * 10
        h = self.encoder(initial_masks)
        h_0 = h.clone()
        # e_shape = (initial_masks.shape[0], initial_masks.shape[1], initial_masks.shape[1] - 1, self.att_din)
        # e = initial_masks[:, :, self.agent_din:].reshape(e_shape)
        # e = self.edge_encoder(e)

        h = self.message_round(h, d_mats, self.rounds)

        h = self.fa(torch.cat([h_0, h], dim=-1))
        h = self.final(h).squeeze(-1)
        y_hat = F.softmax(h, dim=-1).unsqueeze(1)
        return y_hat, h

    def message_round(self, h, d, rounds):

        for i in range(rounds):
            hi, hj = self.i_j(h, d.shape)
            alpha = self.alpha[i](hi, hj, d).unsqueeze(-1)
            e = self.fe[i](hi, hj, d).view(d.shape[0], d.shape[1], d.shape[2], -1)
            m = (alpha * e).sum(dim=-2)
            # h, _ = self.gru(h.view(in_shape), m.view(m_shape))
            # h = h.view(out_shape)
            h = self.fmn(h, m)

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