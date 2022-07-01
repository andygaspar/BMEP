import copy

import torch

from Data_.Dataset.bmep_dataset import BMEP_Dataset
import torch.optim as optim
from torch import nn


from torch.utils.data import DataLoader

from Net.Nets.gnn import GNN

data_ = BMEP_Dataset()
batch_size = 128
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)


# dgn = DGN_(8, 128, 128, 6)
# dgn = GNN(num_inputs=2, h_dimension=512, hidden_dim=512, num_messages=7)
dgn = GNN(num_inputs=2, h_dimension=512, hidden_dim=512, num_messages=7)
# y_hat = dgn.forward(adj_mats[0].unsqueeze(0), d_mats[0].unsqueeze(0), initial_masks[0].unsqueeze(0),
#                     masks[0].unsqueeze(0))

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

optimizer = optim.Adam(dgn.parameters(), lr=1e-5, weight_decay=1e-3)
# optimizer = optim.SGD(dgn.parameters(), lr=1e-4, momentum=0.9)
k, yy = None, None
best_loss = 1e+4
best_net = copy.deepcopy(dgn)
for epoch in range(25_000):
    loss = None
    for data in dataloader:
        adj_mats, d_mats, initial_masks, masks, y = data
        optimizer.zero_grad()
        output, h = dgn(adj_mats, d_mats, initial_masks, masks)
        # out, yy = output[masks > 0], y[masks > 0]
        # loss = criterion(out, yy)
        loss = criterion(output, y.view(y.shape[0], -1))
        loss.backward()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_net = copy.deepcopy(dgn)
        # torch.nn.utils.clip_grad_norm_(dgn.parameters(), max_norm=0.001, norm_type=float('inf'))
        optimizer.step()
    if epoch % 20 == 0:
        with torch.no_grad():

            idx = torch.max(output, dim=-1)[1]
            prediction = torch.zeros_like(output)
            id = torch.tensor(range(idx.shape[0])).to("cuda:0")
            prediction[id, idx] = 1
            err = torch.sum(torch.abs(prediction - y.view(y.shape[0], -1)))
            print(epoch, loss.item(), "error", err.item() / 2, "over", y.shape[0], "  best", best_loss)

        if epoch % 500 == 0:
            with torch.no_grad():

                o = (torch.matmul(h[0], h[0].permute(1, 0)) * masks[0])
                j = list(masks[0].nonzero())
                o = o[[x[0] for x in j], [x[1] for x in j]]
                print(torch.mean(h[0], dim=-2))
                print(torch.std(h[0], dim=-2))
                print(j)
                print(o, y.nonzero()[0], "\n")

best_net.save_weights("Net/Nets/net" + str(int(best_loss * 100)/100))

