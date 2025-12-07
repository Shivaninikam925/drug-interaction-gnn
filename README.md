# Drug–Drug Interaction Prediction Using Graph Neural Networks

This repository implements a graph neural network (GNN) for predicting potential drug–drug interactions (DDIs) using the TWOSIDES dataset.
The project converts drug pairs into lightweight graph structures and trains a GCN-based model to learn patterns associated with adverse interactions.
It is designed as a clear, extensible baseline for future research involving chemical features, molecular graphs, and large-scale pharmacological reasoning.

## Project Overview

Drug–drug interactions can contribute to significant clinical risk, especially in polypharmacy settings.
This work builds a simple but research-oriented pipeline that:

Represents each drug pair as a small graph

Learns relational patterns using graph convolutional networks

Predicts whether a given drug pair is likely to interact

Establishes a baseline for more sophisticated chemical-feature models

The intention is to provide an interpretable, reproducible framework that can support academic research, collaboration, and downstream model development.

## Dataset: TWOSIDES

The TWOSIDES dataset contains:

Drug A identifier (ID1)

Drug B identifier (ID2)

Binary interaction label

Optional side-effect descriptions 

For efficient experimentation, the current workflow processes a 10,000-sample subset, though the code supports the complete dataset.

Reference:
Tatonetti NP et al. Data-driven prediction of drug effects and interactions. Science Translational Medicine, 2012.

## Model Architecture

The model is intentionally simple and serves as a strong baseline:

Learned embeddings for drug identifiers

Two GCNConv layers for message passing

Global mean pooling to obtain graph-level representations

A final linear layer for binary classification

### Architecture Diagram
 Drug A (node) ----\
                     → GCN → ReLU → GCN → GlobalMeanPool → Linear → Sigmoid
 Drug B (node) ----/


This design focuses on relational structure rather than chemical features.

## Training Results

Using 10,000 samples and a basic two-layer GCN:

Loss converges rapidly

Test accuracy reaches approximately 99%
(expected given identifier-only embeddings)

Future work will replace ID embeddings with chemical fingerprints to produce more realistic results.

## Relevant Literature

Tatonetti NP et al. Data-driven prediction of drug effects and interactions. Sci Transl Med, 2012.

Zitnik M, Agrawal M, Leskovec J. Modeling polypharmacy side effects using graph convolutional networks. Bioinformatics, 2018.

Gaudelet T et al. Utilizing graph machine learning within drug discovery and development. Nat Rev Drug Discov, 2021.

## Citation

If you use or reference this project:

@misc{ddi-gnn-2025,
  author       = {Shivani},
  title        = {Drug–Drug Interaction Prediction using Graph Neural Networks},
  year         = {2025},
  note         = {GitHub repository},
  url          = {https://github.com/Shivaninikam925/drug-interaction-gnn}
}

## Contact

If you are a researcher or professor interested in discussing this work or exploring collaboration opportunities, feel free to reach out.

