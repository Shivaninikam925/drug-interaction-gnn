import os
import pickle
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from dataset_loader import TwoSidesDDI
from gnn_model import GNNDrugInteractionModel


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch)              # [batch_size]
        labels = batch.y.view(-1).to(device)

        if preds.dtype != labels.dtype:
            labels = labels.to(preds.dtype)

        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def evaluate(model, loader, device):
    model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            p = model(batch)
            preds_all.extend(p.cpu().tolist())
            labels_all.extend(batch.y.view(-1).cpu().tolist())

    if len(preds_all) == 0:
        return 0.0, float("nan"), float("nan"), [], []

    preds = np.array(preds_all)
    labels = np.array(labels_all)

    acc = accuracy_score(labels, (preds > 0.5).astype(int))
    try:
        roc = roc_auc_score(labels, preds)
    except Exception:
        roc = float("nan")
    try:
        pr = average_precision_score(labels, preds)
    except Exception:
        pr = float("nan")

    return acc, roc, pr, preds_all, labels_all


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # Load / process dataset (expects data/raw/two-sides.csv)
    dataset = TwoSidesDDI(root="data", subset_size=5000)
    print("Dataset size:", len(dataset))

    if len(dataset) < 10:
        print("Not enough data to train. Check two-sides.csv.")
        return

    node_dim = dataset[0].x.size(1)
    total = len(dataset)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    train_ds = dataset[:train_end]
    val_ds = dataset[train_end:val_end]
    test_ds = dataset[val_end:]

    print(f"Train: {len(train_ds)} Val: {len(val_ds)} Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = GNNDrugInteractionModel(node_feature_dim=node_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()

    results = {
        "train_losses": [],
        "val_metrics": [],
        "test_metrics": None,
        "val_preds": None,
        "val_labels": None,
        "test_preds": None,
        "test_labels": None
    }

    epochs = 10
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        acc, roc, pr, val_preds, val_labels = evaluate(model, val_loader, device)

        results["train_losses"].append(loss)
        results["val_metrics"].append((acc, roc, pr))
        results["val_preds"] = val_preds
        results["val_labels"] = val_labels

        print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Acc {acc:.3f} | ROC {roc:.3f} | PR {pr:.3f}")

    # Final test
    acc_t, roc_t, pr_t, test_preds, test_labels = evaluate(model, test_loader, device)
    results["test_metrics"] = (acc_t, roc_t, pr_t)
    results["test_preds"] = test_preds
    results["test_labels"] = test_labels

    print("\nFinal Test Performance:")
    print("Accuracy:", acc_t)
    print("ROC-AUC:", roc_t)
    print("PR-AUC:", pr_t)

    os.makedirs("results", exist_ok=True)
    with open("results/results_summary.pkl", "wb") as f:
        pickle.dump(results, f)
    print("✔ Saved results to results/results_summary.pkl")


if __name__ == "__main__":
    main()
