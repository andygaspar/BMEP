import torch

from Data_.Dataset.bmep_dataset import BMEP_Dataset
from Net.Nets.gnn import DGN
import torch.optim as optim
from torch import nn


from torch.utils.data import DataLoader

from Net.Nets.gnn_working import GNN
from Net.Nets.representation_test.rep_4 import DGN_

data_ = BMEP_Dataset()
batch_size = 64
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)


# dgn = DGN_(8, 128, 128, 6)
dgn = GNN(num_inputs=2, att_inputs=64, hidden_dim=64, num_messages=4)
# y_hat = dgn.forward(adj_mats[0].unsqueeze(0), d_mats[0].unsqueeze(0), initial_masks[0].unsqueeze(0),
#                     masks[0].unsqueeze(0))

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

optimizer = optim.Adam(dgn.parameters(), lr=1e-5, weight_decay=1e-3)
# optimizer = optim.SGD(dgn.parameters(), lr=1e-4, momentum=0.9)
k, yy = None, None
for epoch in range(1_000_000):
    loss = None
    for data in dataloader:
        adj_mats, d_mats, initial_masks, masks, y = data
        optimizer.zero_grad()
        output, k = dgn(adj_mats, d_mats, initial_masks, masks)
        # out, yy = output[masks > 0], y[masks > 0]
        # loss = criterion(out, yy)
        loss = criterion(output, y.view(y.shape[0], -1))


        loss.backward()
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), max_norm=0.001, norm_type=float('inf'))
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), 1)
        optimizer.step()
    if epoch % 20 == 0:
        print(epoch, loss.item())
        if epoch % 100 == 0:

            with torch.no_grad():
                for i in range(10):
                    o = (torch.matmul(k[i], k[i].permute(1, 0)) * masks[0])
                    j = list(masks[i].nonzero())
                    o = o[[x[0] for x in j], [x[1] for x in j]]
                    print(k[i])
                    print(j)
                    print(o, y.nonzero()[i], "\n")

