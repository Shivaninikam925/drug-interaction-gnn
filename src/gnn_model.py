import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class DDIChemGNN(nn.Module):
    """
    Simple GCN-based GNN for molecular interaction prediction.

    - Input: merged drug A + drug B molecular graph
    - Output: probability of interaction in [0,1]
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph-level embedding (one vector per drug pair)
        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))   # probability
        return x.view(-1)