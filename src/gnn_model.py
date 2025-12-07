import torch
from torch.nn import Linear, ReLU, Sigmoid
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNDrugInteractionModel(torch.nn.Module):
    def __init__(self, node_feature_dim):
        super().__init__()

        self.gnn1 = GCNConv(node_feature_dim, 64)
        self.gnn2 = GCNConv(64, 64)

        self.fc1 = Linear(64, 32)
        self.fc2 = Linear(32, 1)

        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = self.relu(self.gnn1(x, edge_index))
        h = self.relu(self.gnn2(h, edge_index))

        graph_emb = global_mean_pool(h, batch)

        h = self.relu(self.fc1(graph_emb))
        out = self.sigmoid(self.fc2(h))

        return out.view(-1)
