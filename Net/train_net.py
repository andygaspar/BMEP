import copy
import shutil
import time
import json

import numpy as np
import torch
import os
from Net.network_manager import NetworkManager

print(os.getcwd())

from Data_.Datasets.bmep_dataset import BMEP_Dataset
import torch.optim as optim
from torch import nn

from torch.utils.data import DataLoader
from importlib.metadata import version

a100 = True if version('torch') == '1.9.0+cu111' else False
edge = False

folder = 'GNN_TAU'
data_folder = '03-M18_5_9'
save = True

net_manager = NetworkManager(folder, supervised=True)
params = net_manager.get_params()
train_params, net_params = params["train"], params["net"]
train_params["train data"] = data_folder

criterion = train_params["criterion"]
cross_entropy = True if criterion == "cross" else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_ = BMEP_Dataset(folder_name=data_folder, scale_d=net_params["scale_d"], start=train_params["start"],
                     end=train_params["end"], a100=a100)
batch_size = train_params["batch_size"]
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)

dgn = net_manager.make_network(normalisation_factor=data_.max_d_mat)





optimizer = optim.Adam(dgn.parameters(), lr=10 ** train_params["lr"], weight_decay=10 ** train_params["weight_decay"])
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
        net_input = data[:-1]
        optimizer.zero_grad()
        output, h = dgn(net_input)
        loss = net_manager.compute_loss(criterion, output, data)

        loss.backward()
        losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            if epoch > 100:
                if directory is not None and save:
                    shutil.rmtree(directory)
                directory = dgn.save_net(folder, best_loss, params)
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), max_norm=0.001, norm_type=float('inf'))
        optimizer.step()
    if epoch % 25 == 0:
        with torch.no_grad():
            idx = torch.max(output, dim=-1)[1]
            if a100 and cross_entropy:
                err = torch.nonzero(idx - y).shape[0]
            else:
                prediction = torch.zeros_like(output)
                id = torch.tensor(range(idx.shape[0])).to(device)
                prediction[id, idx] = 1
                err = torch.sum(torch.abs(prediction - y.view(y.shape[0], -1))) / 2
            net_manager.standings.append([epoch, np.mean(losses), 'last_loss', loss.item(), "error", err, "over",
                                          y.shape[0], "  best", best_loss, 'non one', sum(output[output < 0.9])])
            print(epoch, np.mean(losses), 'last_loss', loss.item(), "error", err, "over", y.shape[0],
                  "  best", best_loss, 'non one', sum(output[output < 0.9]))

            losses = []

        if epoch % 100 == 0:
            net_manager.write_standings()
            with torch.no_grad():
                if not a100:
                    o = (torch.matmul(h[0], h[0].permute(1, 0)) * masks[0])
                    j = list(masks[0].nonzero())
                    o = o[[x[0] for x in j], [x[1] for x in j]]
                    print(torch.mean(h[0], dim=-2))
                    print(torch.std(h[0], dim=-2))
                    print(j)
                    print(o, y.nonzero()[0], "\n")
                else:
                    if edge:
                        print(h[0] * masks[0])
                        print("pred", torch.argmax((h[0] * masks[0]).flatten()).item(), "  y", y[0].item(), "\n")

print("training time", (time.time() - t)/60, 'mins')
