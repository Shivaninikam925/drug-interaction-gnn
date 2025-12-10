import os
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from torch_geometric.data import Data, InMemoryDataset

RDLogger.DisableLog('rdApp.*')


def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        int(atom.GetImplicitValence()),
        int(atom.GetExplicitValence()),
    ], dtype=torch.float)


def mol_to_graph(mol):
    if mol is None:
        return None

    atoms = mol.GetAtoms()
    if len(atoms) == 0:
        return None

    x = torch.stack([atom_features(a) for a in atoms], dim=0)

    edge_index = []
    edge_attr = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = float(b.GetBondTypeAsDouble())
        edge_index += [[i, j], [j, i]]
        edge_attr += [[bt], [bt]]

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return x, edge_index, edge_attr


class TwoSidesDDI(InMemoryDataset):
    def __init__(self, root, subset_size=5000, transform=None, pre_transform=None):
        self.subset_size = subset_size

        # allow safe PyG loading
        try:
            import torch.serialization as ts
            import torch_geometric.data.data as tgdata
            ts.add_safe_globals([tgdata.Data])
        except:
            pass

        super().__init__(root, transform, pre_transform)

        # load processed dataset
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["two-sides.csv"]

    @property
    def processed_file_names(self):
        return ["processed_dataset.pt"]

    def download(self):
        return  # CSV must already exist

    def process(self):
        print("STEP 1: Reading CSV...")
        df = pd.read_csv(self.raw_paths[0])

        df["X1"] = df["X1"].astype(str).str.strip()
        df["X2"] = df["X2"].astype(str).str.strip()
        df["Y"] = df["Y"].apply(lambda v: 1 if float(v) > 0 else 0)

        print("Remaining rows:", len(df))

        # BALANCE: 5000 positives + 5000 negatives
        pos = df[df["Y"] == 1]
        neg_pool = df[df["Y"] == 0]

        pos = pos.sample(n=self.subset_size, random_state=42)
        neg = neg_pool.sample(n=self.subset_size, random_state=42)

        df = pd.concat([pos, neg]).reset_index(drop=True)
        print("Final dataset size:", len(df))

        print("STEP 2: Building graphs...")
        data_list = []
        skipped = 0

        for idx, row in df.iterrows():
            smi1, smi2 = row["X1"], row["X2"]
            mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)

            if mol1 is None or mol2 is None:
                skipped += 1
                continue

            g1 = mol_to_graph(mol1)
            g2 = mol_to_graph(mol2)
            if g1 is None or g2 is None:
                skipped += 1
                continue

            x1, e1, a1 = g1
            x2, e2, a2 = g2
            n1, n2 = x1.size(0), x2.size(0)

            # shift drug2 edges
            e2 = e2 + n1

            x = torch.cat([x1, x2], dim=0)
            edge_index = torch.cat([e1, e2], dim=1)
            edge_attr = torch.cat([a1, a2], dim=0)

            # Morgan fingerprints -----------------------
            try:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
                fp1 = torch.tensor(fp1, dtype=torch.float)
            except:
                fp1 = torch.zeros(2048)

            try:
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
                fp2 = torch.tensor(fp2, dtype=torch.float)
            except:
                fp2 = torch.zeros(2048)
            # ---------------------------------------------

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([row["Y"]], dtype=torch.float),
                split=torch.tensor([n1], dtype=torch.long),
                fp1=fp1,
                fp2=fp2
            )

            data_list.append(data)

        print("Graphs built:", len(data_list))
        print("Skipped rows:", skipped)
        torch.save(self.collate(data_list), self.processed_paths[0])
        print("Done!")
