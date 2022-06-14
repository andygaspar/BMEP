import abc
import copy

from torch import nn
import torch


class Network(nn.Module):
    def __init__(self, ):
        super(Network, self).__init__()
        self.loss = 0

    def get_action(self, state):
        with torch.no_grad():
            state: torch.Tensor
            return self.forward(state.unsqueeze(0))

    def take_weights(self, model_network):
        self.load_state_dict(copy.deepcopy(model_network.state_dict()))

    def load_weights(self, file):
        self.load_state_dict(torch.load(file))

    def save_weights(self, filename: str):
        torch.save(self.state_dict(), filename + '.pt')

    def save_model(self, filename: str):
        model = self.vgg16(pretrained=True)
        torch.save(model.state_dict(), filename + '.pt')