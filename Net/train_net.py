import copy
import shutil
import time
import json

import numpy as np
import torch
import os
print(os.getcwd())

from Data_.Dataset.bmep_dataset import BMEP_Dataset
import torch.optim as optim
from torch import nn

from torch.utils.data import DataLoader

#from Net.Nets.GNN.gnn import GNN
from Net.Nets.GNN1.gnn_1 import GNN_1
from importlib.metadata import version

a100 = True if version('torch') == '1.9.0+cu111' else False

path = 'Net/Nets/GNN1/'
net_name = "GNN_1"
save = True

with open(path + 'params.json', 'r') as json_file:
    params = json.load(json_file)
    print(params)

train_params, net_params = params["train"], params["net"]

dgn = GNN_1(net_params)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_ = BMEP_Dataset(scale_d=net_params["scale_d"], start=train_params["start"], end=train_params["end"], a100=a100)
batch_size = train_params["batch_size"]
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

optimizer = optim.Adam(dgn.parameters(), lr=train_params["lr"], weight_decay=train_params["weight_decay"])
# optimizer = optim.SGD(dgn.parameters(), lr=1e-4, momentum=0.9)
k, yy = None, None
best_loss = 1e+4
best_net = copy.deepcopy(dgn)
losses = []
t = time.time()
directory = None
for epoch in range(train_params["epochs"]):
    loss = None
    for data in dataloader:
        adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, y = data
        optimizer.zero_grad()
        output, h = dgn(adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks)
        # out, yy = output[masks > 0], y[masks > 0]
        # loss = criterion(out, yy)
        loss = criterion(output, y)
        loss.backward()
        losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            if epoch > 100:
                if directory is not None and save:
                    shutil.rmtree(directory)
                directory = dgn.save_net(path, best_loss, net_name, net_params)
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), max_norm=0.001, norm_type=float('inf'))
        optimizer.step()
    if epoch % 25 == 0:
        with torch.no_grad():
            idx = torch.max(output, dim=-1)[1]
            if a100:
                err = torch.nonzero(idx - y).shape[0]
            else:
                prediction = torch.zeros_like(output)
                id = torch.tensor(range(idx.shape[0])).to(device)
                prediction[id, idx] = 1
                err = torch.sum(torch.abs(prediction - y.view(y.shape[0], -1)))
            print(epoch, np.mean(losses), 'last_loss', loss.item(), "error", err.item() / 2, "over", y.shape[0],
                  "  best", best_loss)
            losses = []


        if epoch % 100 == 0:
            with torch.no_grad():
                o = (torch.matmul(h[0], h[0].permute(1, 0)) * masks[0])
                j = list(masks[0].nonzero())
                o = o[[x[0] for x in j], [x[1] for x in j]]
                print(torch.mean(h[0], dim=-2))
                print(torch.std(h[0], dim=-2))
                print(j)
                print(o, y.nonzero()[0], "\n")
print(time.time() - t)




