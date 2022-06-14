import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

# USE_CUDA = torch.cuda.is_available()
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
#
#                                                                                                                 **kwargs)
from network import Network


class Encoder(nn.Module):
    def __init__(self, din=32, hidden_dim=128, device=None):

        super(Encoder, self).__init__()
        self.device = device
        self.fc = nn.Linear(din, hidden_dim).to(self.device)

    def forward(self, x):
        embedding = F.relu(self.fc(x)).to(self.device)
        return embedding


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


class Q_Net(nn.Module):
    def __init__(self, hidden_dim, dout, device):
        self.device = device
        super(Q_Net, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout).to(self.device)

    def forward(self, x):
        q = self.fc(x)
        return q

class TestNet(nn.Module):
    def __init__(self, num_inputs, device):
        self.device = device
        super(TestNet, self).__init__()
        self.fc = nn.Linear(num_inputs*2, 32).to(self.device)
        self.fc2 = nn.Linear(32, 3).to(self.device)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return x


class DGN(Network):
    def __init__(self, n_agent, num_inputs, hidden_dim, num_actions, network=None):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask = torch.ones((n_agent, n_agent)).to(self.device)

        self.encoder = Encoder(num_inputs, hidden_dim, self.device)
        self.att_1 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim, self.device)
        self.att_2 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim, self.device)
        self.att_3 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim, self.device)
        self.q_net = Q_Net(hidden_dim, num_actions, self.device)

        if network is not None:
            self.load_weights(network)
        # self.test_net = TestNet(num_inputs, self.device)

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.att_1(h1, self.mask)
        h3 = self.att_2(h2, self.mask)
        h4 = self.att_2(h3, self.mask)
        q = self.q_net(h4)
        # a = torch.zeros_like(x)
        # a[:, 0, :] = x[:, 1, :]
        # a[:, 1, :] = x[:, 0, :]
        # x = torch.cat([x, a], dim=-1)
        # q = self.test_net(x)
        return q

