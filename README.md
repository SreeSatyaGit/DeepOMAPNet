# DeepOMAPNet: Graph-Attention Multi-Modal Single-Cell Analysis

---

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
  <img src="https://img.shields.io/badge/Focus-Bioinformatics_&_Deep_Learning-blue.svg" alt="Focus">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

**DeepOMAPNet** is a high-performance deep learning framework designed for integrated multi-modal single-cell analysis (CITE-seq). By combining **Graph Attention Networks (GAT)** with **Cross-Modal Transformer Fusion**, DeepOMAPNet achieves superior accuracy in mapping RNA expression to surface protein (ADT) levels, while simultaneously enabling cell-type identification and disease classification (e.g., AML vs. Normal).

---

## ✨ Key Features

- **🧬 Multi-Modal Integration**: Bridging the gap between transcriptomics (RNA) and proteomics (ADT).
- **🕸️ Graph-Based Learning**: Captures cellular heterogeneity by leveraging k-NN graph topologies.
- **⚡ Transformer Fusion**: Advanced cross-modal attention mechanism for robust feature alignment.
- **🎯 Multi-Task Optimization**: Simultaneous training on ADT regression, cell-type classification, and disease diagnosis.
- **📈 Scalable & Efficient**: Sparse attention layers and automatic mixed-precision (AMP) for large-scale CITE-seq datasets.
- **🔄 Transfer Learning**: Efficient adapter-based fine-tuning for cross-dataset application.

---

## 🚀 Installation & Reproducibility

DeepOMAPNet is developed in Python 3.8 and PyTorch. Follow these steps to set up a reproducible environment.

### 1. Requirements
Ensure you have [Conda](https://docs.conda.io/en/latest/) installed.

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/SreeSatyaGit/DeepOMAPNet.git
cd DeepOMAPNet

# Create and activate environment
conda env create -f environment.yml
conda activate deepomapnet
```

Alternatively, install via pip:
```bash
pip install -r requirements.txt
```

---

## 📖 Getting Started

We provide comprehensive tutorials to help you go from raw data to trained models.

### 🏋️ Training
To train the model on your own CITE-seq data:
1.  Navigate to `Tutorials/Training.ipynb`.
2.  Follow the data loading steps (supports `AnnData` objects).
3.  Execute the training pipeline to generate model weights and performance metrics.

### 🧪 Evaluation
Use `Tutorials/Test.ipynb` to evaluate a pre-trained model on unseen datasets, generate UMAP visualizations, and compute correlation metrics (Pearson/Spearman).

### 🔧 Fine-tuning
For transfer learning on new datasets where modalities might be unaligned, refer to `Tutorials/Finetune.ipynb`.

---

## 📂 Repository Structure

| Module | Components |
| :--- | :--- |
| **`scripts/model/`** | `doNET.py` (Core GAT + Transformer architecture), Adapters, and Positional Encodings. |
| **`scripts/data_provider/`** | `data_preprocessing.py` (CLR/Z-score normalization), `graph_data_builder.py` (PyG Graph conversion). |
| **`scripts/trainer/`** | `gat_trainer.py` (Main training/eval loop), `fineTune.py` (Transfer learning logic). |
| **`Tutorials/`** | Interactive Jupyter Notebooks for end-to-end workflows. |
| **`R/`** | Supporting R scripts for WNN mapping and legacy preprocessing. |
| **`publication_figures/`** | Placeholder for analytical plots and published result figures. |

---

## 📐 Architecture Overview

DeepOMAPNet's architecture is built on four pillars:
1.  **GAT Encoder**: Learns 96-dimensional hidden representations from the gene expression k-NN graph.
2.  **Positional Encoding**: Integrates graph statistics (degree, clustering coefficients) into the latent space.
3.  **Cross-Modal Transformer**: Uses multi-head attention to fuse RNA and ADT information.
4.  **Multi-Task Heads**: Specialized MLP branches for regression (ADT) and classification (AML/Cell-Type).

Detailed diagram: [model_architecture.html](model_architecture.html)

---

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

---
Developed by **DeepOMAPNet Contributors** | [Link to Paper](https://github.com/SreeSatyaGit/DeepOMAPNet)
