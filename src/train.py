import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from dataset_loader import TwoSidesDDI
from gnn_model import GNNDrugInteractionModel


def split_dataset(dataset):
    n = len(dataset)
    return dataset[:int(n*0.8)], dataset[int(n*0.8):int(n*0.9)], dataset[int(n*0.9):]


def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(-1)

        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            preds.extend(pred.cpu().numpy())
            labels.extend(batch.y.view(-1).cpu().numpy())

    preds_bin = [1 if p > 0.5 else 0 for p in preds]

    acc = accuracy_score(labels, preds_bin)
    try:
        roc = roc_auc_score(labels, preds)
    except:
        roc = float("nan")
    pr = average_precision_score(labels, preds)

    return acc, roc, pr


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # LOAD DATA
    dataset = TwoSidesDDI(root="data", subset_size=5000)
    train_set, val_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=32)
    test_loader  = DataLoader(test_set, batch_size=32)

    node_dim = dataset[0].x.size(1)
    print("Node feature dim:", node_dim)

    model = GNNDrugInteractionModel(node_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(1, 11):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        acc, roc, pr = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {acc:.4f} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")

    print("\nFinal Test:")
    acc, roc, pr = evaluate(model, test_loader, device)
    print("Accuracy:", acc)
    print("ROC-AUC:", roc)
    print("PR-AUC:", pr)


if __name__ == "__main__":
    main()
