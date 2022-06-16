import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

# USE_CUDA = torch.cuda.is_available()
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
#
#                                                                                                                 **kwargs)
from Net.network import Network

softmax = nn.functional.softmax


class LeafLayer(nn.Module):
    def __init__(self, m, hidden_dim=128, device=None):
        super(LeafLayer, self).__init__()
        self.device = device
        self.eta_f = nn.Sequential(
            nn.Linear(m, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m)
        ).to(self.device)

    def forward(self, A, d, h):
        h = torch.matmul(d, h) / (torch.max(d.view(d.shape[0], 1, -1), dim=-1)[0] * 6)
        return A, d, self.eta_f(h)


class NodesLayer(nn.Module):
    def __init__(self, m, hidden_dim=128, device=None):
        super(NodesLayer, self).__init__()
        self.device = device
        self.eta_g = nn.Sequential(
            nn.Linear(m, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m)
        ).to(self.device)

    def forward(self, A, d, h):
        h = torch.matmul(A, h) / 3
        return A, d, self.eta_g(h)


class AttModel(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout, device):
        self.device = device
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim).to(self.device)
        self.fck = nn.Linear(din, hidden_dim).to(self.device)
        self.fcq = nn.Linear(din, hidden_dim).to(self.device)
        self.fcout = nn.Linear(hidden_dim, dout).to(self.device)

    def forward(self, x, mask):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0, 2, 1)
        att = F.softmax(torch.mul(torch.bmm(q, k), mask) - 9e15 * (1 - mask), dim=2)

        out = torch.bmm(att, v)
        # out = torch.add(out,v)
        out = F.relu(self.fcout(out))
        return out


class DGN(Network):
    def __init__(self, m, hidden_dim_f, hidden_dim_g, n_messages, composed=True, network=None):
        super().__init__()
        self.m = m
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        leaf_layers = [LeafLayer(m, hidden_dim_f, self.device)] * n_messages
        nodes_layers = [NodesLayer(m, hidden_dim_g, self.device)] * n_messages
        self.t_l = torch.normal(0, 1, size=(1, m), dtype=torch.float).to(self.device)
        self.t_f = torch.normal(0, 1, size=(1, m), dtype=torch.float).to(self.device)

        self.layers = [val for pair in zip(leaf_layers, nodes_layers) for val in pair] if composed \
            else leaf_layers + nodes_layers
        # self.net = nn.Sequential(*layers)

        if network is not None:
            self.load_weights(network)
        # self.test_net = TestNet(num_inputs, self.device)

    def forward(self, A, d, x, mask):
        h = self.t_l * x[:, :, -2].view((x.shape[0], -1, 1)) + self.t_f * x[:, :, -1].view((x.shape[0], -1, 1))
        # h = self.net(A, d, h0)
        for l in self.layers:
            _, _, h = l(A, d, h)

        y_hat = torch.matmul(h, h.permute(0, 2, 1))
        mask[mask == 0] = -torch.inf
        y_hat = y_hat * mask
        y = y_hat.view((h.shape[0], -1))
        y_hat = softmax(y_hat.view((h.shape[0], -1)), dim=-1)
        return y_hat
