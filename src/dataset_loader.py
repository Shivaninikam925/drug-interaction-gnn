import os
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
import torch_geometric


class TwoSidesDDI(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, subset_size=None):
        self.subset_size = subset_size
        super().__init__(root, transform, pre_transform)

        # Allow PyG custom classes for PyTorch 2.6+
        import torch.serialization
        torch.serialization.add_safe_globals([torch_geometric.data.Data])

        # Load processed file (includes num_drugs)
        self.data, self.slices, self.num_drugs = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def raw_file_names(self):
        return ["two-sides.csv"]  

    @property
    def processed_file_names(self):
        return ["twosides_graph.pt"]

    def download(self):
        pass

    def process(self):
        raw_path = self.raw_paths[0]
        df = pd.read_csv(raw_path)

        print("Processing...")

        # Optional subset for speed
        if self.subset_size is not None:
            df = df.sample(self.subset_size, random_state=42)

        # TWO-SIDES COLUMN NAMES:
        # ['ID1', 'ID2', 'Y', 'Side Effect Name', 'X1', 'X2']

        # Standardize labels into 0/1
        df["Y"] = df["Y"].apply(lambda x: 1 if x > 0 else 0)

        # Building integer mapping for drug IDs
        all_ids = pd.concat([df["ID1"], df["ID2"]]).unique()
        id_map = {drug_id: idx for idx, drug_id in enumerate(all_ids)}

        # Saving for embedding layer
        self.num_drugs = len(id_map)

        data_list = []

        for _, row in df.iterrows():
            drug1 = id_map[row["ID1"]]
            drug2 = id_map[row["ID2"]]
            label = float(row["Y"])

            # Node features
            x = torch.tensor([[drug1], [drug2]], dtype=torch.long)

            # 2-node fully connected graph
            adj = torch.tensor([[0, 1],
                                [1, 0]], dtype=torch.float)

            edge_index = dense_to_sparse(adj)[0]

            data = Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor([label], dtype=torch.float)
            )

            data_list.append(data)

        data, slices = self.collate(data_list)

        # SAVE 
        torch.save(
            (data, slices, self.num_drugs),
            self.processed_paths[0]
        )

        print("Done!")