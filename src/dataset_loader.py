import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        1 if atom.GetIsAromatic() else 0,
        int(atom.GetImplicitValence()),
        int(atom.GetExplicitValence()),
    ], dtype=torch.float)


def mol_to_graph_data_obj(mol):
    if mol is None:
        return None

    atoms = mol.GetAtoms()
    if len(atoms) == 0:
        return None

    x = torch.stack([atom_features(a) for a in atoms])

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = float(bond.GetBondTypeAsDouble())
        edge_index += [[i, j], [j, i]]
        edge_attr += [[bt], [bt]]

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return x, edge_index, edge_attr


class TwoSidesDDI(InMemoryDataset):
    def __init__(self, root, subset_size=5000):
        self.subset_size = subset_size
        super().__init__(root)

        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["two-sides.csv"]

    @property
    def processed_file_names(self):
        return ["processed_dataset.pt"]

    def process(self):
        print("STEP 1: Reading CSV...")
        df = pd.read_csv(self.raw_paths[0])
        print("Total rows:", len(df))

        df["X1"] = df["X1"].astype(str).str.strip()
        df["X2"] = df["X2"].astype(str).str.strip()
        df = df[~df["X1"].isin(["nan", ""]) & ~df["X2"].isin(["nan", ""])]
        print("Remaining rows:", len(df))

        df["Y"] = df["Y"].apply(lambda v: 1 if float(v) > 0 else 0)

        if len(df) > self.subset_size:
            df = df.sample(self.subset_size, random_state=42)

        df_pos = df[df["Y"] == 1].reset_index(drop=True)
        print("Positives selected:", len(df_pos))

        # SAFE NEGATIVE SAMPLING
        smiles_map = {}
        for _, r in df_pos.iterrows():
            smiles_map[r["ID1"]] = r["X1"]
            smiles_map[r["ID2"]] = r["X2"]

        ids = list(smiles_map.keys())
        neg_rows = []

        while len(neg_rows) < len(df_pos):
            d1, d2 = np.random.choice(ids, 2, replace=False)
            neg_rows.append([d1, d2, 0, smiles_map[d1], smiles_map[d2]])

        df_neg = pd.DataFrame(neg_rows, columns=["ID1", "ID2", "Y", "X1", "X2"])

        # ✅ CRITICAL FIX: SHUFFLE BEFORE SPLIT
        df_final = pd.concat([df_pos, df_neg], ignore_index=True)\
                      .sample(frac=1.0, random_state=42)\
                      .reset_index(drop=True)

        print("Final dataset size:", len(df_final))

        print("STEP 2: Building graphs...")
        data_list = []

        for _, row in df_final.iterrows():
            mol1, mol2 = Chem.MolFromSmiles(row["X1"]), Chem.MolFromSmiles(row["X2"])
            if mol1 is None or mol2 is None:
                continue

            g1, g2 = mol_to_graph_data_obj(mol1), mol_to_graph_data_obj(mol2)
            if g1 is None or g2 is None:
                continue

            x1, e1, a1 = g1
            x2, e2, a2 = g2
            n1 = x1.size(0)

            if e2.numel() > 0:
                e2 = e2 + n1

            x = torch.cat([x1, x2])
            e = torch.cat([e1, e2], dim=1) if e1.numel() or e2.numel() else torch.zeros((2, 0), dtype=torch.long)
            a = torch.cat([a1, a2]) if a1.numel() or a2.numel() else torch.zeros((0, 1))

            data = Data(
                x=x,
                edge_index=e,
                edge_attr=a,
                y=torch.tensor([row["Y"]], dtype=torch.float),
                split=torch.tensor([n1], dtype=torch.long)
            )
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
        print("Done!")
