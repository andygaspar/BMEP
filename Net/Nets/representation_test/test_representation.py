import numpy as np
from scipy import special
import torch

from Data_.Dataset.bmep_dataset import BMEP_Dataset
from Net.Nets.gnn import DGN
import torch.optim as optim
from torch import nn


from torch.utils.data import DataLoader

from Net.Nets.representation_test.rep_4 import DGN_
from Net.Nets.representation_test.rep_5 import DGN_5
from Net.Nets.representation_test.representation_net import DGN_test
from Net.Nets.representation_test.representation_net_2 import DGN_test_2
import matplotlib.pyplot as plt

from Net.Nets.representation_test.representation_net_3 import DGN_test_3


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            # print(n, p)
            # print(p.grad)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.show()


batch_size = 64
message_rounds = 3

data_ = BMEP_Dataset()
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)


dgn = DGN_5(num_inputs=2, att_inputs=16, hidden_dim=16)
# y_hat = dgn.forward(adj_mats[0].unsqueeze(0), d_mats[0].unsqueeze(0), initial_masks[0].unsqueeze(0),
#                     masks[0].unsqueeze(0))
# print(dgn.leaf_layers[0].fck.state_dict()['weight'])
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

optimizer = optim.Adam(dgn.parameters(), lr=1e-4, weight_decay=1e-3)
# optimizer = optim.SGD(dgn.parameters(), lr=1e-4, momentum=0.9)
k, yy = None, None
t = torch.tensor([[[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]] for _ in range(batch_size)], dtype=torch.float32).to("cuda:0")
for epoch in range(1_000_000):
    loss = None
    for data in dataloader:
        adj_mats, d_mats, initial_masks, masks, y = data
        optimizer.zero_grad()
        output, h = dgn(adj_mats, d_mats, initial_masks, masks)

        # out, yy = output[masks>0], y[masks>0]
        # loss = criterion(output.view(y.shape[0], -1), y.view(y.shape[0], -1))
        loss = criterion(t[:output.shape[0]], output)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), max_norm=0.001, norm_type=float('inf'))
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), 1)
        optimizer.step()

    if epoch % 20 == 0:
        print(epoch, loss.item())
        # print(dgn.leaf_layers[0].fck.state_dict()['weight'])
        if epoch % 100 == 0 and epoch > 0:

            with torch.no_grad():
                print(h[0])
                print(output[0])

            # plot_grad_flow(dgn.named_parameters())

import numpy as np
from itertools import combinations
# A = data_.adj_mats[1].cpu().numpy()
# T = np.copy(A).astype(int)
# idx = np.where(A[0, :] == 1)
#
# for i in range(6,10):
#     idx = np.where(A[i, :] == 1)[0]
#     print(idx)
#     for c in list(combinations(idx, 2)):
#         print(c)
#         T[c] = 2
#         T[c[::-1]] = 2
#
#
# z =np.array([[0, 3, 9, 8, 8, 9, 4, 6, 5, 8, 3, 3],
#                             [3, 0, 10, 9, 9, 10, 5, 7, 6, 9, 2, 4],
#                             [9, 10, 0, 5, 5, 2, 7, 5, 6, 3, 10, 8],
#                             [8, 9, 5, 0, 2, 5, 6, 4, 5, 4, 9, 7],
#                             [8, 9, 5, 2, 0, 5, 6, 4, 5, 4, 9, 7],
#                             [9, 10, 2, 5, 5, 0, 7, 5, 6, 3, 10, 8],
#                             [4, 5, 7, 6, 6, 7, 0, 4, 3, 6, 5, 3],
#                             [6, 7, 5, 4, 4, 5, 4, 0, 3, 4, 7, 5],
#                             [5, 6, 6, 5, 5, 6, 3, 3, 0, 5, 6, 4],
#                             [8, 9, 3, 4, 4, 3, 6, 4, 5, 0, 9, 7],
#                             [3, 2, 10, 9, 9, 10, 5, 7, 6, 9, 0, 4],
#                             [3, 4, 8, 7, 7, 8, 3, 5, 4, 7, 4, 0]])
#
# from Pardi.pardi import Pardi
#
#
# p = Pardi(z)
# A = p.adj_mats[-1]
# T = np.copy(A).astype(int)
#
# for i in range(12, 22):
#     for j in range(i+1 ,22):
#         found = False
#         while not found:
#             for k in np.where(T[i, i:]==1):
#                 if k ==

# d = data_.d_mats[0].cpu().numpy()
#
# v = np.ones(10)
# soft_d = special.softmax(d, axis=0)
# for _ in range(50):
#     v = np.dot(soft_d, v)
#     v = v/sum(v)
#     print(v)
#
# v

