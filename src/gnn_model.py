import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNDrugInteractionModel(nn.Module):
    """
    Expects each Data to have:
      - x: concatenated nodes (drug1 then drug2)
      - edge_index: edges for concatenated graph
      - split: tensor([n_nodes_drug1]) per sample
    Works correctly with PyG DataLoader batching (uses ptr + per-sample split).
    """

    def __init__(self, node_feature_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def encode_local(self, x_local, edge_index_local, batch_local):
        """Run 2-layer GCN + mean pooling on local subgraph (edge_index_local uses local indexing)."""
        h = torch.relu(self.conv1(x_local, edge_index_local))
        h = torch.relu(self.conv2(h, edge_index_local))
        return global_mean_pool(h, batch_local)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        ptr = data.ptr       # length = batch_size + 1
        splits = data.split.view(-1)  # per-sample number of nodes in first molecule

        device = x.device
        hidden_1 = []
        hidden_2 = []
        B = ptr.size(0) - 1

        for i in range(B):
            s = int(ptr[i].item())
            e = int(ptr[i + 1].item())
            n1 = int(splits[i].item())

            # local node features for this sample [s:e)
            x_local = x[s:e]

            # edges inside this sample, rebased to local indexing
            if edge_index.numel() == 0:
                e_local = torch.zeros((2, 0), dtype=torch.long, device=device)
            else:
                mask = (edge_index[0] >= s) & (edge_index[0] < e) & (edge_index[1] >= s) & (edge_index[1] < e)
                if mask.sum() == 0:
                    e_local = torch.zeros((2, 0), dtype=torch.long, device=device)
                else:
                    e_local = edge_index[:, mask] - s

            # drug1: nodes 0..n1-1 (local)
            if n1 > 0:
                if e_local.numel() == 0:
                    e1 = torch.zeros((2, 0), dtype=torch.long, device=device)
                else:
                    e1_mask = (e_local[0] < n1) & (e_local[1] < n1)
                    e1 = e_local[:, e1_mask]
                batch1 = torch.zeros(n1, dtype=torch.long, device=device)
                h1 = self.encode_local(x_local[:n1], e1, batch1)
            else:
                h1 = torch.zeros((1, self.conv2.out_channels), device=device)

            # drug2: nodes n1..(local_n-1)
            n_local = x_local.size(0)
            n2 = n_local - n1
            if n2 > 0:
                if e_local.numel() == 0:
                    e2 = torch.zeros((2, 0), dtype=torch.long, device=device)
                else:
                    e2_mask = (e_local[0] >= n1) & (e_local[1] >= n1)
                    e2 = e_local[:, e2_mask] - n1
                batch2 = torch.zeros(n2, dtype=torch.long, device=device)
                h2 = self.encode_local(x_local[n1:], e2, batch2)
            else:
                h2 = torch.zeros((1, self.conv2.out_channels), device=device)

            hidden_1.append(h1)
            hidden_2.append(h2)

        h1_all = torch.cat(hidden_1, dim=0)  # [B, hidden]
        h2_all = torch.cat(hidden_2, dim=0)  # [B, hidden]

        pair = torch.cat([h1_all, h2_all], dim=1)  # [B, 2*hidden]
        out = self.mlp(pair).view(-1)  # [B]
        return out
