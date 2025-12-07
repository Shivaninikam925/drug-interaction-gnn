import os
import random
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
import torch_geometric
import torch.serialization

# ---- Allow PyTorch 2.6 to load PyG Data objects safely ----
torch.serialization.add_safe_globals([
    torch_geometric.data.Data,
])


def generate_negative_samples(df_pos: pd.DataFrame, num_negatives: int) -> pd.DataFrame:
    """
    Create negative (non-interaction) drug pairs that do NOT appear
    in the positive interaction set.

    Returns a DataFrame with the same columns: ID1, ID2, Y, Side Effect Name, X1, X2.
    For negatives, X1/X2 are set to 'NONE' and will be replaced by a dummy
    molecule ('C') later.
    """
    print("Generating negative samples...")

    # Unique drug IDs from the positive set
    unique_drugs = list(set(df_pos["ID1"]).union(set(df_pos["ID2"])))

    # All known positive pairs (order-independent)
    positive_pairs = set(
        tuple(sorted((row["ID1"], row["ID2"])))
        for _, row in df_pos.iterrows()
    )

    negative_pairs = set()

    # Sample until we have the desired number of negatives
    while len(negative_pairs) < num_negatives:
        d1, d2 = random.sample(unique_drugs, 2)
        pair = tuple(sorted((d1, d2)))
        if pair not in positive_pairs:
            negative_pairs.add(pair)

    print(f"Generated {len(negative_pairs)} negative samples.")

    neg_df = pd.DataFrame(list(negative_pairs), columns=["ID1", "ID2"])
    neg_df["Y"] = 0  # label = non-interaction
    neg_df["Side Effect Name"] = "none"
    neg_df["X1"] = "NONE"
    neg_df["X2"] = "NONE"

    return neg_df


def mol_to_graph(mol: Chem.Mol) -> Data | None:
    """Convert an RDKit Mol into a simple PyG graph with atom-number features."""
    if mol is None:
        return None

    # Node features: here just atomic number → shape [num_atoms, 1]
    x = torch.tensor(
        [[atom.GetAtomicNum()] for atom in mol.GetAtoms()],
        dtype=torch.float,
    )

    # Adjacency → edge_index (undirected)
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.tensor(adj, dtype=torch.float)
    edge_index = dense_to_sparse(adj)[0]

    return Data(x=x, edge_index=edge_index)


class TwoSidesDDI(InMemoryDataset):
    """
    TWOSIDES-based Drug–Drug Interaction dataset with RDKit graphs + negative sampling.

    Each example is a merged molecular graph of:
      - Drug A (from SMILES X1)
      - Drug B (from SMILES X2)
    with a binary label:
      - 1.0 => interaction
      - 0.0 => sampled non-interaction
    """

    def __init__(self, root: str, transform=None, pre_transform=None,
                 subset_size: int | None = 5000, include_negatives: bool = True):
        self.subset_size = subset_size
        self.include_negatives = include_negatives
        super().__init__(root, transform, pre_transform)

        # Important: weights_only=False because this is not a pure weight checkpoint
        self.data, self.slices = torch.load(
            self.processed_paths[0],
            weights_only=False
        )

    @property
    def raw_file_names(self):
        # Expect data/raw/two-sides.csv
        return ["two-sides.csv"]

    @property
    def processed_file_names(self):
        return ["twosides_mol_graphs.pt"]

    def download(self):
        # CSV is assumed to be already present in data/raw
        pass

    def process(self):
        print("Processing TWOSIDES into RDKit molecular graphs...")

        df_raw = pd.read_csv(self.raw_paths[0])

        # 1) Positive examples
        if self.subset_size is not None and self.subset_size < len(df_raw):
            df_pos = df_raw.sample(self.subset_size, random_state=42).reset_index(drop=True)
        else:
            df_pos = df_raw.copy().reset_index(drop=True)

        # 2) Negative examples via sampling
        if self.include_negatives:
            df_neg = generate_negative_samples(df_pos, num_negatives=len(df_pos))
            df = pd.concat([df_pos, df_neg], ignore_index=True)
        else:
            df = df_pos

        print("Final dataset size (pos + neg):", len(df))

        data_list: list[Data] = []

        for _, row in df.iterrows():
            # Handle SMILES, including negative samples ("NONE")
            smiles1 = row["X1"]
            smiles2 = row["X2"]

            if pd.isna(smiles1) or smiles1 == "NONE" or str(smiles1).strip() == "":
                smiles1 = "C"   # dummy one-atom molecule
            if pd.isna(smiles2) or smiles2 == "NONE" or str(smiles2).strip() == "":
                smiles2 = "C"

            mol1 = Chem.MolFromSmiles(str(smiles1))
            mol2 = Chem.MolFromSmiles(str(smiles2))

            if mol1 is None or mol2 is None:
                continue

            g1 = mol_to_graph(mol1)
            g2 = mol_to_graph(mol2)

            if g1 is None or g2 is None:
                continue

            # Merge the two drug graphs into one
            offset = g1.x.size(0)
            x = torch.cat([g1.x, g2.x], dim=0)
            edge_index = torch.cat(
                [g1.edge_index, g2.edge_index + offset],
                dim=1
            )

            # Binary label: >0 => interaction, 0 => sampled negative
            label_val = 1.0 if row["Y"] > 0 else 0.0

            data = Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor([label_val], dtype=torch.float)
            )

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        print(f"Molecular graph processing complete! Samples: {len(data_list)}")
        print("Done!")