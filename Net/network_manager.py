import json

from Net.network import Network
from Net.Nets.GNN.gnn import GNN
from Net.Nets.GNN1.gnn_1 import GNN_1
from Net.Nets.GNN2.gnn_2 import GNN_2
from Net.Nets.GNN_edge.gnn_edge import GNN_edge
from Net.Nets.GNNGRU.gnn_gru import GNN_GRU

nets_dict = {
    'GNN': GNN,
    'GNN_1': GNN_1,
    'GNN_2': GNN_2,
    'GNN_edge': GNN_edge,
    'GNNGRU': GNN_GRU
}


class NetworkManager:

    def make_network(self, folder):
        path = 'Net/Nets/' + folder + '/'
        with open(path + 'params.json', 'r') as json_file:
            params = json.load(json_file)
            print(params)

        train_params, net_params = params["train"], params["net"]

        dgn = nets_dict[folder](net_params=net_params)

        return dgn, train_params, net_params, folder

    @staticmethod
    def get_network(folder, file):
        path = 'Net/Nets/' + folder + '/' + file + '/'

        with open(path + 'params.json', 'r') as json_file:
            params = json.load(json_file)
            print(params)
        net_params = params

        dgn = nets_dict[folder](net_params=net_params, network=path + "weights.pt")
        return dgn
