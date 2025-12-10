import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNDrugInteractionModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim=64):
        super().__init__()

        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        fp_dim = 2048  # Morgan fingerprint dimension

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2 * fp_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def encode(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(h, batch)  # [1, hidden_dim]

    def forward(self, data):
        x, edge_index, ptr = data.x, data.edge_index, data.ptr
        splits = data.split.view(-1)
        B = ptr.size(0) - 1

        h1_list, h2_list = [], []

        for i in range(B):
            start = ptr[i].item()
            end = ptr[i+1].item()
            n1 = splits[i].item()
            mid = start + n1

            mask = (edge_index[0] >= start) & (edge_index[0] < end)
            mask &= (edge_index[1] >= start) & (edge_index[1] < end)
            edges_local = edge_index[:, mask] - start

            x_local = x[start:end]

            # drug1 edges
            e1_mask = (edges_local[0] < n1) & (edges_local[1] < n1)
            e1 = edges_local[:, e1_mask]
            h1 = self.encode(x_local[:n1], e1)

            # drug2 edges
            e2_mask = (edges_local[0] >= n1) & (edges_local[1] >= n1)
            e2 = edges_local[:, e2_mask] - n1
            h2 = self.encode(x_local[n1:], e2)

            h1_list.append(h1)
            h2_list.append(h2)

        h1_all = torch.cat(h1_list, dim=0)
        h2_all = torch.cat(h2_list, dim=0)

        fp1 = data.fp1.to(x.device)
        fp2 = data.fp2.to(x.device)

        fused = torch.cat([h1_all, h2_all, fp1, fp2], dim=1)

        return self.mlp(fused).view(-1)
