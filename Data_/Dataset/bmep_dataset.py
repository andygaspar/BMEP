import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BMEP_Dataset(Dataset):

    def __init__(self, scale_d=1, start=0, end=None, a100=False, tau=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.d_mats = torch.load("Data_/Dataset/d_mats.pt").to(torch.float).to(device)[start: end]
        self.d_mats /= torch.max(self.d_mats).item() * scale_d
        self.d_masks = torch.load("Data_/Dataset/d_masks.pt").to(torch.float).to(device)[start: end]
        self.initial_masks = torch.load("Data_/Dataset/initial_masks.pt").to(torch.float).to(device)[start: end]
        self.adj_mats = torch.load("Data_/Dataset/adj_mats.pt").to(torch.float).to(device)[start: end]
        self.ad_masks = torch.load("Data_/Dataset/ad_masks.pt").to(torch.float).to(device)[start: end]
        self.masks = torch.load("Data_/Dataset/masks.pt").to(torch.float).to(device)[start: end]
        self.y = torch.load("Data_/Dataset/y.pt").to(torch.float).to(device)[start: end]
        self.tau = None if not tau else self.compute_taus()
        if a100:
            self.y = torch.nonzero(self.y.view(self.y.shape[0], -1))[:, 1]
        else:
            self.y = self.y.view(self.y.shape[0], -1)
        self.size_masks = torch.ones_like(self.adj_mats)
        self.size = self.d_mats.shape[0]

    def __getitem__(self, index):
        return self.adj_mats[index], self.ad_masks[index], self.d_mats[index], self.d_masks[index],\
               self.size_masks[index], self.initial_masks[index],  self.masks[index], self.y[index]

    def __len__(self):
        return self.size

    def compute_taus(self):
        taus = []
        for adj in self.adj_mats:
            g = nx.from_numpy_matrix(adj.to("cpu").numpy())
            Tau = nx.floyd_warshall_numpy(g)
            Tau[np.isinf(Tau)] = 0
            taus.append(Tau)

        self.tau = torch.tensor(taus)

        torch.save(self.tau, "Data_/Dataset/taus.pt")

