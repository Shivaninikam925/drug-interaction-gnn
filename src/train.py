import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset_loader import TwoSidesDDI
from gnn_model import DDIChemGNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binarize_labels(y_tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure labels are strictly 0/1.
    (Dataset already stores 0/1, but this is a safe guard.)
    """
    return (y_tensor > 0).float()


def get_data_loaders(root: str = "data", subset_size: int = 5000,
                     batch_size: int = 32):
    dataset = TwoSidesDDI(root=root, subset_size=subset_size, include_negatives=True)

    print("Total graphs:", len(dataset))

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)                     # [batch_size]
        targets = binarize_labels(batch.y).view(-1).to(device)

        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)

        probs = model(batch)   # [batch_size], already sigmoid
        targets = binarize_labels(batch.y).view(-1).to(device)

        all_probs.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Accuracy at 0.5 threshold
    preds = (all_probs >= 0.5).astype("int32")
    acc = (preds == all_targets).mean()

    # ROC-AUC and PR-AUC
    try:
        roc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        roc = float("nan")

    try:
        pr = average_precision_score(all_targets, all_probs)
    except ValueError:
        pr = float("nan")

    return acc, roc, pr


def main():
    train_loader, val_loader, test_loader, dataset = get_data_loaders(
        root="data",
        subset_size=5000,
        batch_size=32,
    )

    in_channels = dataset[0].x.size(1)
    print("Node feature dim:", in_channels)

    model = DDIChemGNN(in_channels=in_channels, hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()

    epochs = 10

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_acc, val_roc, val_pr = evaluate(model, val_loader)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val ROC-AUC: {val_roc:.4f} | "
            f"Val PR-AUC: {val_pr:.4f}"
        )

    # Final test evaluation
    test_acc, test_roc, test_pr = evaluate(model, test_loader)
    print("\nFinal Test Metrics:")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"ROC-AUC  : {test_roc:.4f}")
    print(f"PR-AUC   : {test_pr:.4f}")


if __name__ == "__main__":
    main()