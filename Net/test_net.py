import json

import numpy as np
import torch

from Data_.Dataset.bmep_dataset import BMEP_Dataset

from torch.utils.data import DataLoader

from Net.Nets.GNN.gnn import GNN

path = 'Net/Nets/GNN/_3.66/'

with open(path + 'params.json', 'r') as json_file:
    params = json.load(json_file)
    print(params)
train_params, net_params = params["train"], params["net"]

data_ = BMEP_Dataset()
batch_size = train_params["batch_size"]
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)


dgn = GNN(net_params=net_params, network=path + "weights.pt")


# d = data_.d_mats[0]
# net_solver = NetSolver(d, dgn)
# net_solver.solve()
# print(net_solver.solution)

errors = []
lenghts = []
for epoch in range(10):
    for data in dataloader:
        with torch.no_grad():
            adj_mats, d_mats, initial_masks, masks, y = data
            output, h = dgn(adj_mats, d_mats, initial_masks, masks)
            idx = torch.max(output, dim=-1)[1]
            prediction = torch.zeros_like(output)
            id = torch.tensor(range(idx.shape[0])).to("cuda:0")
            prediction[id, idx] = 1
            err = torch.sum(torch.abs(prediction - y.view(y.shape[0], -1)))
            errors.append(err.item()/2)
            lenghts.append(y.shape[0])
            print(epoch, "error", err.item() / 2, "over", y.shape[0])

print("mean num errors", np.mean(errors))
print("error rate ", np.sum(errors)/np.sum(lenghts))




