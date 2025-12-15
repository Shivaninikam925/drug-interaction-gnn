# A Graph Neural Network Pipeline for Drug–Drug Interaction Prediction

## Authors
Shiva — (your name)  
Advisor / helper: Nova (assistant) — code and pipeline support

## Date
(put your submission date here)

---

## 1. Introduction

Drug–drug interactions (DDIs) are clinically important and predicting them computationally can reduce adverse events and accelerate pharmacovigilance. Graph neural networks (GNNs) are a promising approach because molecules are naturally represented as graphs (atoms as nodes, bonds as edges). This project builds an end-to-end DDI prediction pipeline that takes paired SMILES strings, builds molecular graphs with RDKit, and trains a GNN classifier on drug pairs.

Goals:
- Build a robust, crash-free data pipeline (SMILES cleaning → RDKit graphs).
- Train a baseline pairwise GNN.
- Analyze dataset properties and report interpretable baseline results.
- Provide a reproducible codebase and clear next steps to substantially improve performance.

---

## 2. Dataset

Source: `data/raw/two-sides.csv` (your file). The CSV is expected to include columns:
- `ID1`, `ID2` — identifiers (drug IDs)
- `X1`, `X2` — SMILES for drug1 and drug2
- `Y` — interaction label (positive if > 0)

Important cleaning steps applied:
- Force SMILES columns to strings and strip whitespace.
- Drop rows where `X1` or `X2` are invalid (strings "nan", "none", "", etc.).
- Convert `Y` to binary: any positive numeric value -> 1, else 0.
- Subsample to a manageable `subset_size` (we used 5000 positives and generated 5000 negatives → total 10000).
- Hard negative sampling: random pairs of drugs not in positive set (ensures negatives exist).

**Label distribution (final dataset used in experiments)**:
- Train positives: 4000 / 8000
- Val positives: 500 / 1000
- Test positives: 500 / 1000

(We generated negatives so labels are balanced in the final dataset used for the baseline model.)

---

## 3. Data processing & graph construction

- RDKit used to parse SMILES (`Chem.MolFromSmiles`). RDKit warnings are suppressed in code.
- Atom-level features (per-node, small, simple):
  - atomic number
  - degree
  - formal charge
  - total hydrogens
  - aromatic flag
  - implicit valence
  - explicit valence
- Bond encoding: single floating feature = bond type (`GetBondTypeAsDouble()`), expanded later.
- For each data row (drug1, drug2):
  - Build `Data` with concatenated `x` (x1 || x2), `edge_index` (edges for both graphs with indices rebased for second graph), `edge_attr` (edges attributes concatenated), `y` (label), and `split` (number of nodes in drug1).
- Save processed dataset with PyG `InMemoryDataset` format.

---

## 4. Model architecture

Baseline: pairwise GCN encoder + MLP classifier

- Two-layer GCN per subgraph (shared parameters).
- Global mean pooling to obtain one vector per drug.
- Concatenate two drug vectors → MLP → single sigmoid output (probability of interaction).

Hyperparameters:
- Node feature dim: 7 (as above)
- GCN hidden dim: 64
- Optimizer: Adam, lr = 1e-3
- Loss: BCELoss (sigmoid final layer)
- Batch size: 32
- Epochs: 10

---

## 5. Training details

- Train/Val/Test split: 80% / 10% / 10%.
- Balanced dataset used by generating negatives.
- Training loop ensures correct shapes: `preds` is `[batch_size]`, `labels` is `[batch_size]`.
- Evaluate Accuracy (threshold 0.5), ROC-AUC, and PR-AUC.
- Safe handling of edge cases where ROC/PR undefined.

---

## 6. Representative results

From representative run on dataset of 10k pairs (5k positives, 5k generated negatives):

**Validation / Test metrics (final):**
- Test Accuracy: **0.566**
- Test ROC-AUC: **0.607**
- Test PR-AUC: **0.590**

**Observed behavior & notes**
- Baseline GCN with only atom-level small features demonstrates limited performance — expected for a minimal baseline.
- Adding bond features and Morgan fingerprints consistently improves performance in literature.
- When dataset was used *without negatives*, label imbalance made training trivial (predict-all-ones) with meaningless metrics — we solved this by generating negatives.

---

## 7. Diagnostic analysis & dataset issues

We observed earlier runs where almost all labels were positive (no negatives present); model then trivially predicted the positive class. The key lesson: **always check label distribution** before training.

We add recommended diagnostics:
- Print label counts after processing and per-split counts.
- Print `y` unique values and `np.bincount`.
- Visualize SMILES parsing failure rate.

---

## 8. Figures to include (and script)

Recommended figures:
1. Label distribution (train / val / test)
2. Training loss curve (loss vs epoch)
3. Validation ROC curve & PR curve
4. Distribution of predicted probabilities (histogram)
5. Example molecule pair visualizations (RDKit drawing) — show a couple positive and negative pairs

Below are plotting scripts (save as `plot_results.py`). They assume you saved evaluation outputs during training (or can be adapted to run evaluate functions on `test_loader` and store arrays).

```python
# plot_results.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pickle
import json

# --- LOAD saved predictions (you can dump preds and labels from evaluate function) ---
# Expect a JSON or pickle with keys: preds_val, labels_val, preds_test, labels_test, train_loss_per_epoch
with open('results_summary.pkl', 'rb') as f:
    rs = pickle.load(f)

preds_val = np.array(rs['preds_val'])
labels_val = np.array(rs['labels_val'])
preds_test = np.array(rs['preds_test'])
labels_test = np.array(rs['labels_test'])
train_loss = np.array(rs.get('train_loss', []))

# 1) Label distribution
plt.figure(figsize=(6,4))
vals = [labels_val.sum(), (labels_val==0).sum()]
plt.bar(['positive', 'negative'], vals)
plt.title('Validation label distribution')
plt.savefig('fig_label_dist_val.png', dpi=200, bbox_inches='tight')

# 2) Loss curve
if train_loss.size > 0:
    plt.figure(figsize=(6,4))
    plt.plot(train_loss, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    plt.title('Training loss per epoch')
    plt.savefig('fig_train_loss.png', dpi=200, bbox_inches='tight')

# 3) ROC & PR curves (validation)
fpr, tpr, _ = roc_curve(labels_val, preds_val)
roc_auc = auc(fpr, tpr)
prec, recall, _ = precision_recall_curve(labels_val, preds_val)
pr_auc = auc(recall, prec)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Validation ROC Curve')
plt.legend()
plt.savefig('fig_val_roc.png', dpi=200, bbox_inches='tight')

plt.figure(figsize=(6,5))
plt.plot(recall, prec, label=f'PR (AUC={pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Validation Precision-Recall Curve')
plt.legend()
plt.savefig('fig_val_pr.png', dpi=200, bbox_inches='tight')

# 4) Predicted probability histogram
plt.figure(figsize=(6,4))
plt.hist(preds_test, bins=50)
plt.title('Predicted probability distribution (test)')
plt.savefig('fig_test_pred_hist.png', dpi=200, bbox_inches='tight')

print("Saved figures: fig_label_dist_val.png, fig_train_loss.png, fig_val_roc.png, fig_val_pr.png, fig_test_pred_hist.png")
