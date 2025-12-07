import torch
from torch_geometric.loader import DataLoader
from dataset_loader import TwoSidesDDI
from gnn_model import GNNDrugInteractionModel
from sklearn.model_selection import train_test_split
import numpy as np

# Load Dataset
print("Loading TWOSIDES dataset...")

dataset = TwoSidesDDI(root="data", subset_size=10000)
print("Dataset size:", len(dataset))

# Train-test split
indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = dataset[train_idx]
test_dataset = dataset[test_idx]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize Model
num_drugs = dataset.num_drugs

model = GNNDrugInteractionModel(
    num_drugs=num_drugs,
    embed_dim=32,
    hidden_dim=64
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCELoss()

# Train 
def train():
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        out = model(batch)  # shape: [batch_size]
        y = batch.y.view(-1)  # ensure same shape

        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# Evaluation
def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch)
            preds = (preds > 0.5).float()

            labels = batch.y.view(-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    return correct / total

# Training
for epoch in range(1, 11):
    loss = train()
    acc = evaluate()
    print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

print("Training complete!")