import torch
from torch.utils.data import Dataset, DataLoader
import os


class BMEP_Dataset(Dataset):

    def __init__(self, data_dir, start=0, end=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.d_mats = torch.load(os.path.join(data_dir, "Data_/Dataset/d_mats.pt").to(torch.float).to(device)[start: end]
        self.d_mats /= torch.max(self.d_mats).item()
        self.initial_masks = torch.load(os.path.join(data_dir, "Data_/Dataset/initial_masks.pt")).to(torch.float).to(device)[start: end]
        self.adj_mats = torch.load(os.path.join(data_dir,"Data_/Dataset/adj_mats.pt")).to(torch.float).to(device)[start: end]
        self.masks = torch.load(os.path.join(data_dir,"Data_/Dataset/masks.pt")).to(torch.float).to(device)[start: end]
        self.y = torch.load(os.path.join(data_dir,"Data_/Dataset/y.pt")).to(torch.float).to(device)[start: end]
        self.size = self.d_mats.shape[0]


    def __getitem__(self, index):
        return self.adj_mats[index], self.d_mats[index], self.initial_masks[index],  self.masks[index], self.y[index]

    def __len__(self):
        return self.size
