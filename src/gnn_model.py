import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNDrugInteractionModel(nn.Module):
    def __init__(self, num_drugs, embed_dim=64, hidden_dim=64):
        super().__init__()

        # 1) Learn embeddings for each drug
        self.embedding = nn.Embedding(num_drugs, embed_dim)

        # 2) GNN layers
        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 3) Output layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Map integer drug IDs → embeddings
        x = self.embedding(x.squeeze().long())

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        x = global_mean_pool(x, batch)

        out = self.classifier(x).view(-1)

        return out