import torch

d_mats = torch.load("Data_/Dataset/d_mats.pt")
initial_masks = torch.load( "Data_/Dataset/initial_masks.pt")
adj_mats = torch.load( "Data_/Dataset/adj_mats.pt")
masks = torch.load( "Data_/Dataset/masks.pt")
y = torch.load("Data_/Dataset/y.pt")
