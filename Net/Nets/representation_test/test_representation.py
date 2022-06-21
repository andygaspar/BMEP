import torch

from Data_.Dataset.bmep_dataset import BMEP_Dataset
from Net.Nets.gnn import DGN
import torch.optim as optim
from torch import nn


from torch.utils.data import DataLoader

from Net.Nets.representation_test.representation_net import DGN_test
from Net.Nets.representation_test.representation_net_2 import DGN_test_2

data_ = BMEP_Dataset()
dataloader = DataLoader(dataset=data_, batch_size=64, shuffle=True)


dgn = DGN_test_2(8, 128, 128, 6)
# y_hat = dgn.forward(adj_mats[0].unsqueeze(0), d_mats[0].unsqueeze(0), initial_masks[0].unsqueeze(0),
#                     masks[0].unsqueeze(0))

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

optimizer = optim.Adam(dgn.parameters(), lr=1e-2, weight_decay=1e-3)
# optimizer = optim.SGD(dgn.parameters(), lr=1e-4, momentum=0.9)
k, yy = None, None
for epoch in range(1_000_000):
    loss = None
    for data in dataloader:
        adj_mats, d_mats, initial_masks, masks, y = data
        optimizer.zero_grad()
        output, h = dgn(adj_mats, d_mats, initial_masks, masks)
        # out, yy = output[masks>0], y[masks>0]
        # loss = criterion(output.view(y.shape[0], -1), y.view(y.shape[0], -1))
        loss = criterion(adj_mats, output)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), max_norm=0.001, norm_type=float('inf'))
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), 1)
        optimizer.step()
    if epoch % 20 == 0:
        print(epoch, loss.item())
        if epoch % 100 == 0:

            with torch.no_grad():
                print(h[0])
                print(output[0])
                res = torch.round(output[0]) - adj_mats[0]
                print(res)
                print(torch.sum(torch.abs(res)))


