import numpy as np
import torch
from Data_.Datasets.bmep_dataset import BMEP_Dataset
from torch.utils.data import DataLoader
from ML.Net.network_manager import NetworkManager


folder = 'GNN2'
file = '_3.64'
data_folder = '6_taxa_0'

net_manager = NetworkManager(folder, file)
dgn = net_manager.get_network()

data_ = BMEP_Dataset(folder_name=data_folder)
batch_size = 2000
dataloader = DataLoader(dataset=data_, batch_size=batch_size, shuffle=True)



# d = data_.d_mats[0]
# net_solver = NetSolvers(d, dgn)
# net_solver.solve()
# print(net_solver.solution)

errors = []
lenghts = []

for data in dataloader:
    with torch.no_grad():
        adj_mats, ad_masks, d_mats, d_masks, size_masks, initial_masks, masks, y = data
        output, h = dgn(data)
        step_num = torch.sum(adj_mats, dim=(1, 2)) / 2
        idx = torch.max(output, dim=-1)[1]
        prediction = torch.zeros_like(output)
        id = torch.tensor(range(idx.shape[0])).to("cuda:0")
        prediction[id, idx] = 1
        error_vect = torch.abs(prediction - y.view(y.shape[0], -1))
        step_error = step_num[torch.unique(torch.nonzero(error_vect)[:, 0])]
        b = step_error.unique(return_counts=True)
        num_err = torch.sum(torch.abs(prediction - y.view(y.shape[0], -1)))
        errors.append(num_err.item()/2)
        lenghts.append(y.shape[0])
        print("error", num_err.item() / 2, "over", y.shape[0], b[1])

print("mean num errors", np.mean(errors))
print("error rate ", np.sum(errors)/np.sum(lenghts))




