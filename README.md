# DeepOMAPNet: Multi-Task GAT-Transformer for Cross-Modal Single-Cell Analysis

A graph-aware multi-modal neural network that jointly predicts single-cell protein (ADT) abundances and AML status from RNA profiles, demonstrating strong cross-cohort generalization and biologically consistent gene–protein interactions.

## Overview

DeepOMAPNet combines Graph Attention Networks (GAT) with Transformer fusion layers to perform two key tasks simultaneously:
- **ADT Regression**: Predict protein marker abundances from RNA expression
- **AML Classification**: Classify cells as AML vs. Normal/Healthy

The model achieves state-of-the-art performance through cross-modal attention mechanisms and graph-aware processing of single-cell data.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ disk space for datasets

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/DeepOMAPNet.git
cd DeepOMAPNet
```

2. **Create and activate conda environment**
```bash
conda create -n deepomapnet python=3.8
conda activate deepomapnet
```

3. **Install dependencies**
```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install scanpy anndata pandas numpy scipy scikit-learn
pip install matplotlib seaborn plotly
pip install shap shapely
pip install umap-learn
pip install tqdm

# Optional: for advanced visualizations
pip install networkx
pip install leidenalg
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import scanpy; print(f'Scanpy: {scanpy.__version__}')"
```

## 📊 Data Preparation

### Download Datasets

The model requires single-cell RNA-seq and ADT (protein) data. Supported datasets include:

1. **GSE116256**: AML single-cell dataset
2. **GSM3587990**: Control samples
3. **GSM6805326**: Additional AML cohort

### Data Structure

Place your datasets in the following structure:
```
DeepOMAPNet/
├── data/
│   ├── GSE116256/
│   │   ├── RNA_data.h5ad
│   │   └── ADT_data.h5ad
│   ├── GSM3587990/
│   │   ├── RNA_data.h5ad
│   │   └── ADT_data.h5ad
│   └── GSM6805326/
│       ├── RNA_data.h5ad
│       └── ADT_data.h5ad
```

### Data Format Requirements

- **RNA data**: AnnData object with genes as variables (`adata.var_names`) and cells as observations (`adata.obs_names`)
- **ADT data**: AnnData object with protein markers as variables and cells as observations
- **Sample labels**: Store in `adata.obs['samples']` with format "AML####" or "Control####"

## 🏃‍♂️ Running the Model

### 1. Basic Training

```python
import sys
sys.path.append('scripts')

from data_provider.data_preprocessing import prepare_train_test_anndata
from trainer.gat_trainer import train_gat_transformer_fusion
import torch

# Load and preprocess data
data = prepare_train_test_anndata()
rna_adata, rna_test, adt_adata, adt_test = data

# Create AML labels from sample names
aml_labels = (adt_adata.obs['samples'].str.startswith('AML')).astype(int)

# Training configuration
training_config = {
    'epochs': 100,
    'learning_rate': 0.001,
    'hidden_channels': 32,
    'num_heads': 2,
    'num_attention_heads': 2,
    'num_layers': 1,
    'dropout_rate': 0.6,
    'use_mixed_precision': True,
    'early_stopping_patience': 20,
    'adt_weight': 1.0,
    'classification_weight': 1.0
}

# Train the model
trained_model, rna_data_with_masks, adt_data_with_masks, training_history, adt_mean, adt_std, node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt = train_gat_transformer_fusion(
    rna_data=rna_pyg_data,
    adt_data=adt_pyg_data,
    aml_labels=aml_labels_array,
    rna_anndata=rna_adata,
    adt_anndata=adt_adata,
    **training_config
)
```

### 2. Cross-Cohort Validation (LOCO)

```python
from scripts.cross_cohort_evaluation import run_loco_validation

# Run Leave-One-Cohort-Out validation
loco_results = run_loco_validation(
    cohorts=['GSE116256', 'GSM3587990', 'GSM6805326'],
    training_config=training_config,
    n_seeds=5
)

print(f"LOCO AUROC: {loco_results['aml_auroc']:.3f}")
print(f"LOCO ADT R²: {loco_results['adt_r2']:.3f}")
```

### 3. Model Evaluation

```python
from scripts.model_evaluation import evaluate_model_performance

# Evaluate on test set
results = evaluate_model_performance(
    model=trained_model,
    rna_data=rna_data_with_masks,
    adt_data=adt_data_with_masks,
    aml_labels=aml_labels,
    adt_mean=adt_mean,
    adt_std=adt_std
)

print(f"AML Classification AUROC: {results['aml_auroc']:.3f}")
print(f"ADT Regression R²: {results['adt_r2']:.3f}")
```

## 📈 Visualization and Analysis

### Generate Publication Figures

```python
from scripts.publication_figures import create_all_figures

# Create all figures for the paper
figures = create_all_figures(
    model=trained_model,
    rna_data=rna_data_with_masks,
    adt_data=adt_data_with_masks,
    results=results,
    save_dir='figures/'
)
```

### Individual Visualizations

```python
from scripts.visualizations import (
    plot_umap_fused_embeddings,
    plot_adt_marker_correlations,
    plot_aml_confusion_matrix,
    plot_gene_protein_relationships
)

# UMAP of fused embeddings
plot_umap_fused_embeddings(
    fused_embeddings=results['fused_embeddings'],
    cell_labels=aml_labels,
    save_path='figures/umap_fused_embeddings.png'
)

# ADT marker correlations
plot_adt_marker_correlations(
    adt_true=results['adt_true'],
    adt_pred=results['adt_pred'],
    marker_names=adt_adata.var_names,
    save_path='figures/adt_correlations.png'
)

# AML confusion matrix
plot_aml_confusion_matrix(
    aml_true=results['aml_true'],
    aml_pred=results['aml_pred'],
    save_path='figures/aml_confusion_matrix.png'
)
```

## 🔬 Advanced Features

### Attention Analysis

```python
from scripts.attention_analysis import analyze_attention_patterns

# Extract and analyze attention weights
attention_results = analyze_attention_patterns(
    model=trained_model,
    rna_data=rna_data_with_masks,
    adt_data=adt_data_with_masks
)

# Visualize cross-modal attention
attention_results.plot_cross_modal_attention()
```

### SHAP Analysis

```python
from scripts.shap_analysis import compute_shap_values

# Compute SHAP values for interpretability
shap_results = compute_shap_values(
    model=trained_model,
    rna_data=rna_data_with_masks,
    adt_labels=aml_labels,
    n_samples=1000
)

# Plot SHAP summary
shap_results.plot_summary()
```

## 📁 Project Structure

```
DeepOMAPNet/
├── scripts/
│   ├── model/
│   │   ├── doNET.py              # Main model architecture
│   │   └── transformer_models.py # Transformer fusion layers
│   ├── trainer/
│   │   └── gat_trainer.py        # Training loop and evaluation
│   ├── data_provider/
│   │   ├── data_preprocessing.py # Data loading and preprocessing
│   │   ├── gse116256_loader.py   # GSE116256 dataset loader
│   │   ├── gsm3587990_loader.py # GSM3587990 dataset loader
│   │   └── graph_data_builder.py # Graph construction utilities
│   └── visualizations.py         # Visualization functions
├── Tutorials/
│   ├── Training.ipynb           # Main training tutorial
│   ├── Test.ipynb               # Testing and evaluation
│   └── GAT_Transformer_Fusion_Training.ipynb # Advanced tutorial
├── R/                          # R scripts for data processing
├── figures/                    # Generated figures
├── models/                     # Saved model checkpoints
└── data/                       # Dataset storage
```

## ⚙️ Configuration

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_channels` | 32 | Hidden dimension size |
| `num_heads` | 2 | Number of GAT attention heads |
| `num_attention_heads` | 2 | Number of transformer attention heads |
| `num_layers` | 1 | Number of transformer layers |
| `dropout_rate` | 0.6 | Dropout probability |
| `learning_rate` | 0.001 | Learning rate |
| `epochs` | 100 | Number of training epochs |
| `adt_weight` | 1.0 | Weight for ADT regression loss |
| `classification_weight` | 1.0 | Weight for AML classification loss |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_mixed_precision` | True | Use automatic mixed precision |
| `early_stopping_patience` | 20 | Early stopping patience |
| `train_fraction` | 0.8 | Fraction of data for training |
| `val_fraction` | 0.1 | Fraction of data for validation |

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce model size
   training_config['hidden_channels'] = 16
   training_config['num_heads'] = 1
   
   # Enable CPU fallback
   training_config['use_cpu_fallback'] = True
   ```

2. **Data Loading Errors**
   ```python
   # Check data paths
   import os
   print(os.path.exists('data/GSE116256/RNA_data.h5ad'))
   
   # Verify AnnData format
   import scanpy as sc
   adata = sc.read_h5ad('data/GSE116256/RNA_data.h5ad')
   print(adata.shape, adata.obs.columns)
   ```

3. **Mixed Precision Errors**
   ```python
   # Disable mixed precision
   training_config['use_mixed_precision'] = False
   ```

### Performance Optimization

1. **Memory Management**
   - Use `torch.cuda.empty_cache()` between training runs
   - Reduce batch size or subsample data
   - Enable gradient checkpointing

2. **Training Speed**
   - Use mixed precision training
   - Increase learning rate
   - Reduce model complexity

## 📊 Expected Results

### Performance Benchmarks

- **AML Classification**: AUROC > 0.90 on held-out cohorts
- **ADT Regression**: R² > 0.70 on held-out cohorts
- **Cross-Cohort Generalization**: < 10% performance drop between cohorts

### Computational Requirements

- **Training Time**: 2-4 hours on V100 GPU
- **Memory Usage**: 8-16GB GPU memory
- **Model Size**: ~2.7M parameters

## 📚 Citation

If you use DeepOMAPNet in your research, please cite:

```bibtex
@article{deepomapnet2024,
  title={Learning cross-modal gene–protein programs for AML with a multi-task GAT–Transformer},
  author={Your Name and Collaborators},
  journal={Nature Computational Science},
  year={2024}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Single-cell data from GEO/SRA
- PyTorch Geometric for graph neural networks
- Scanpy for single-cell analysis
- The single-cell genomics community

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Email: your.email@institution.edu
- Documentation: [Link to docs]

---

**Last updated**: October 2024  
**Version**: 1.0.0
