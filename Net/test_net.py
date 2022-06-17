import torch
from Data_.Dataset.bmep_dataset import BMEP_Dataset
from Net.gnn import DGN
import torch.optim as optim
from torch import nn


from torch.utils.data import Dataset, DataLoader


data_ = BMEP_Dataset()
dataloader = DataLoader(dataset=data_, batch_size=30, shuffle=True)



dgn = DGN(4, 64, 64, 3)
# y_hat = dgn.forward(adj_mats[0].unsqueeze(0), d_mats[0].unsqueeze(0), initial_masks[0].unsqueeze(0),
#                     masks[0].unsqueeze(0))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(dgn.parameters(), lr=1e-4, weight_decay=1e-3)
# optimizer = optim.SGD(dgn.parameters(), lr=1e-4, momentum=0.9)
# torch.nan_to_num()

for epoch in range(2):
    for data in dataloader:
        adj_mats, d_mats, initial_masks, masks, y = data
        optimizer.zero_grad()
        output = dgn(adj_mats, d_mats, initial_masks, masks)
        loss = criterion(output, y.view(y.shape[0], -1))
        dgn.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), max_norm=0.001, norm_type=float('inf'))
        dgn.float()
        optimizer.step()
        print(loss.item())

