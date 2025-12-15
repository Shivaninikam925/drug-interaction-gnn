import pickle

with open("results_summary.pkl", "rb") as f:
    data = pickle.load(f)

print("\n--- KEYS IN FILE ---")
print(data.keys())

print("\n--- TRAIN LOSSES (len={}): ---".format(len(data["train_losses"])))
print(data["train_losses"])

print("\n--- VAL METRICS (per epoch): ---")
for i, m in enumerate(data["val_metrics"], 1):
    acc, roc, pr = m
    print(f"Epoch {i:02d} | Acc={acc:.4f} | ROC={roc:.4f} | PR={pr:.4f}")

print("\n--- FINAL TEST METRICS ---")
print(data["test_metrics"])

print("\n--- FIRST 50 VAL PREDS ---")
print(data["val_preds"][:50])

print("\n--- FIRST 50 VAL LABELS ---")
print(data["val_labels"][:50])

print("\n--- FIRST 50 TEST PREDS ---")
print(data["test_preds"][:50])

print("\n--- FIRST 50 TEST LABELS ---")
print(data["test_labels"][:50])
