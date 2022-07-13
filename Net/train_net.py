import copy
import time
import pickle

import numpy as np
import torch

from Data_.Dataset.bmep_dataset import BMEP_Dataset
import torch.optim as optim
from torch import nn


from torch.utils.data import DataLoader

from Net.Nets.gnn import GNN
from Net.Nets.gnn_1 import GNN_1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_ = BMEP_Dataset("/m100/home/userexternal/fcamero1/bmep/BMEP")
batch_size = 128

dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)


# dgn = DGN_(8, 128, 128, 6)
# dgn = GNN(num_inputs=2, h_dimension=512, hidden_dim=512, num_messages=7)
dgn = GNN_1(num_inputs=2, h_dimension=64, hidden_dim=64, num_messages=7)
# y_hat = dgn.forward(adj_mats[0].unsqueeze(0), d_mats[0].unsqueeze(0), initial_masks[0].unsqueeze(0),
#                     masks[0].unsqueeze(0))

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

optimizer = optim.Adam(dgn.parameters(), lr=1e-5, weight_decay=1e-3)
# optimizer = optim.SGD(dgn.parameters(), lr=1e-4, momentum=0.9)
k, yy = None, None
best_loss = 1e+10
best_net = 0

losses = []
t = time.time()
for epoch in range(10000):
    for data in dataloader:
        adj_mats, d_mats, initial_masks, masks, y = data
        optimizer.zero_grad()
        output, h = dgn(adj_mats, d_mats, initial_masks, masks)
        # out, yy = output[masks > 0], y[masks > 0]
        # loss = criterion(out, yy)
        y = y.view(y.shape[0], -1).argmax(dim=-1)
        loss = criterion(output, y.long())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), max_norm=0.001, norm_type=float('inf'))
        optimizer.step()

    loss = 0
    if epoch % 50 == 0:
        with torch.no_grad():
            for data in dataloader:
                adj_mats, d_mats, initial_masks, masks, y = data
                output, h = dgn(adj_mats, d_mats, initial_masks, masks)
                y = y.view(y.shape[0], -1).argmax(dim=-1)
                loss += criterion(output, y.long()).item()
            loss = np.mean(loss)
            losses.append(loss)
            if loss < best_loss:
                best_net = epoch

            dgn.save_weights(f"/m100/home/userexternal/fcamero1/bmep/BMEP/Net/Nets/dumps/net_epoch{epoch:05d}_loss{(loss*100) / 100:.5f}")
print(time.time() - t)
print(f"Best net at epoch {best_net}")

with open(f"/m100/home/userexternal/fcamero1/bmep/BMEP/Net/Nets/dumps/losses.pkl", 'wb') as f:
    pickle.dump(losses, f)
