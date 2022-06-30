import copy

import numpy as np
import torch

from Data_.Dataset.bmep_dataset import BMEP_Dataset
import torch.optim as optim
from torch import nn


from torch.utils.data import DataLoader

from Net.Nets.gnn_1 import GNN
from Net.Nets.gnn_2 import GNN_2

data_ = BMEP_Dataset()
batch_size = 128
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)


# dgn = DGN_(8, 128, 128, 6)
# dgn = GNN(num_inputs=2, h_dimension=512, hidden_dim=512, num_messages=7)
dgn = GNN_2(num_inputs=2, h_dimension=512, hidden_dim=512, num_messages=7, network="Net/Nets/net3.68.pt")
# y_hat = dgn.forward(adj_mats[0].unsqueeze(0), d_mats[0].unsqueeze(0), initial_masks[0].unsqueeze(0),
#                     masks[0].unsqueeze(0))


# optimizer = optim.SGD(dgn.parameters(), lr=1e-4, momentum=0.9)
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




