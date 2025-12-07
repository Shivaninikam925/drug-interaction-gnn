import os
import torch
import pandas as pd
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx
import networkx as nx
import random


def mol_to_graph(mol, drug_flag):
    """Convert an RDKit Mol into a NetworkX graph with node features."""
    G = nx.Graph()

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        feat = [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            drug_flag
        ]
        G.add_node(idx, x=torch.tensor(feat, dtype=torch.float))

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        G.add_edge(u, v)

    return G


def combine_graphs(G1, G2):
    """Merge drug1 and drug2 graphs into a single graph."""
    return nx.disjoint_union(G1, G2)


class TwoSidesDDI(InMemoryDataset):
    def __init__(self, root, subset_size=5000, transform=None, pre_transform=None):
        self.subset_size = subset_size
        super().__init__(root, transform, pre_transform)

        # Load processed dataset
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["two-sides.csv"]

    @property
    def processed_file_names(self):
        return ["twosides_rdkit.pt"]

    def process(self):
        print("Processing...")
        df = pd.read_csv(self.raw_paths[0])

        # Drop rows with missing SMILES
        df = df.dropna(subset=["X1", "X2"])

        # Reduce dataset size
        df = df.sample(self.subset_size, random_state=42)

        # --- NEGATIVE SAMPLING (uses valid SMILES) ---
        pos_pairs = df[["X1", "X2"]].values.tolist()
        smiles_pool = list(df["X1"]) + list(df["X2"])

        neg_samples = set()
        while len(neg_samples) < len(df):
            a, b = random.sample(smiles_pool, 2)
            if [a, b] not in pos_pairs:
                neg_samples.add((a, b))

        neg_df = pd.DataFrame(list(neg_samples), columns=["X1", "X2"])
        neg_df["Y"] = 0

        df_pos = df.copy()
        df_pos["Y"] = 1

        # Combine positive + negative
        df_all = pd.concat([df_pos, neg_df], ignore_index=True)
        df_all = df_all.sample(frac=1, random_state=42)

        print(f"Final dataset size: {len(df_all)}")

        data_list = []

        for _, row in df_all.iterrows():
            smi1 = row["X1"]
            smi2 = row["X2"]
            y = row["Y"]

            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)

            if mol1 is None or mol2 is None:
                continue  # skip invalid SMILES

            G1 = mol_to_graph(mol1, drug_flag=0)
            G2 = mol_to_graph(mol2, drug_flag=1)

            G = combine_graphs(G1, G2)

            pyg = from_networkx(G)

            # Convert list of tensors → tensor matrix
            pyg.x = torch.stack([feat for feat in pyg.x], dim=0)

            pyg.y = torch.tensor([y], dtype=torch.float)

            data_list.append(pyg)

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])
        print("RDKit graph processing complete.")
