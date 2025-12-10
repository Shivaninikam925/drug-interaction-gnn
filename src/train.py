import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from dataset_loader import TwoSidesDDI
from gnn_model import GNNDrugInteractionModel


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0
    count = 0

    for batch in loader:
        batch = batch.to(device)

        preds = model(batch)
        labels = batch.y.view(-1).to(device)

        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        count += 1

    return total / count


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            p = model(batch).cpu().tolist()
            l = batch.y.view(-1).cpu().tolist()

            preds += p
            labels += l

    preds = torch.tensor(preds).numpy()
    labels = torch.tensor(labels).numpy()

    acc = accuracy_score(labels, (preds > 0.5).astype(int))

    try:
        roc = roc_auc_score(labels, preds)
    except:
        roc = float("nan")

    try:
        pr = average_precision_score(labels, preds)
    except:
        pr = float("nan")

    return acc, roc, pr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    dataset = TwoSidesDDI(root="data", subset_size=5000)
    print("Dataset size:", len(dataset))

    node_dim = dataset[0].x.size(1)

    N = len(dataset)
    train_ds = dataset[: int(0.8*N)]
    val_ds   = dataset[int(0.8*N): int(0.9*N)]
    test_ds  = dataset[int(0.9*N):]

    print("\nTrain:", len(train_ds))
    print("Val:  ", len(val_ds))
    print("Test: ", len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32)
    test_loader  = DataLoader(test_ds, batch_size=32)

    model = GNNDrugInteractionModel(node_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(1, 11):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        acc, roc, pr = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Acc {acc:.4f} | ROC {roc} | PR {pr}")

    print("\nFINAL TEST RESULTS:")
    acc, roc, pr = evaluate(model, test_loader, device)
    print("Accuracy:", acc)
    print("ROC-AUC:", roc)
    print("PR-AUC:", pr)


if __name__ == "__main__":
    main()
