# Drug–Drug Interaction Prediction using Graph Neural Networks

## Overview
Drug–drug interactions (DDIs) pose significant risks in clinical settings, often leading to adverse drug reactions and treatment failure. Predicting DDIs computationally can reduce experimental costs and improve patient safety.

This project implements a Graph Neural Network (GNN)–based pipeline to predict drug–drug interactions from molecular structure data. Each drug molecule is represented as a graph derived from its SMILES string, and a pairwise GNN model is trained to classify whether two drugs interact.

The focus of this work is on clean dataset construction, safe negative sampling, and robust evaluation, rather than achieving state-of-the-art performance.

## Key Ideas

(1)Represent drug molecules as graphs (atoms as nodes, bonds as edges)

(2)Encode each drug using a Graph Convolutional Network (GCN)

(3)Learn interaction patterns by combining embeddings of two drugs

(4)Handle extreme class imbalance using safe, balanced negative sampling

(5)Evaluate using ROC-AUC and PR-AUC, which are appropriate for imbalanced data

## Dataset

>> Source: TwoSides Drug–Drug Interaction Dataset

>> Raw size: ~4.6 million rows

>> Processed subset: ~10,000 drug pairs (balanced positives & negatives)

Label Processing:

Original interaction counts were converted to binary labels:

 >> Y > 0 → 1 (interaction present)

 >> Y = 0 → 0 (no interaction)

Negative Sampling Strategy:

To avoid label noise and invalid molecules:

>> Negatives are generated only from drugs appearing in positive samples

>> Ensures all SMILES strings are valid and parsable

>> Produces a balanced dataset suitable for supervised learning

## Model Architecture

>> Node features: Basic atomic properties (atomic number, degree, valence, aromaticity, etc.)

>> Encoder: Two-layer Graph Convolutional Network (GCN)

>> Pooling: Global mean pooling per molecule

>> Classifier: MLP over concatenated embeddings of two drugs

>> Loss: Binary Cross-Entropy (BCE)

## Results (Baseline)

| Metric   | Value |
| -------- | ----- |
| Accuracy | ~0.55 |
| ROC-AUC  | ~0.59 |
| PR-AUC   | ~0.56 |

These results indicate that the model learns a non-random interaction signal, despite using simple features and a lightweight architecture.

## Project Structure

drug-interaction-gnn/
│
├── data/
│   ├── raw/                # Original CSV (two-sides.csv)
│   └── processed/          # Cached PyG dataset
│
├── src/
│   ├── dataset_loader.py   # Data cleaning, graph construction, sampling
│   ├── gnn_model.py        # GNN architecture
│   └── train.py            # Training & evaluation loop
│
├── results/
│   └── results_summary.pkl # Saved metrics & predictions
│
└── README.md

## Limitations

>> Uses simple atom features (no fingerprints or pretrained embeddings)

>> GCN baseline only (no attention or message-passing enhancements)

>> Subsampled dataset for computational feasibility

>> These limitations are intentional to establish a clean, interpretable baseline.

## Future Work

>> Incorporate bond features and edge embeddings

>> Add Morgan fingerprints or pretrained molecular embeddings

>> Explore Graph Attention Networks (GATs)

>> Train on larger subsets or full dataset

>> Extend to multi-class interaction types

## Why This Project Matters

This project demonstrates:

>> End-to-end ML research workflow

>> Careful data handling in noisy biomedical datasets

>> Correct evaluation for imbalanced classification

>> Strong foundation for further research in computational biology and ML

## Author

Shivani Nikam
Second-year B.Tech CSE student 
Interests: Machine Learning, Graph Neural Networks, Computational Biology, Research

