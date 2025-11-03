# DeepOMAPNet - File Structure Guide



## 📁 Root Directory

- **`README.md`** - Main documentation with installation, usage, and API reference
- **`LICENSE`** - License information
- **`requirements.txt`** - Python package dependencies
- **`environment.yml`** - Conda environment configuration
- **`model_architecture.html`** - Visual representation of the model architecture

## 📂 scripts/ - Core Python Modules

### Model Architecture (`scripts/model/`)

- **`doNET.py`** - Core model implementation
  - `GATWithTransformerFusion`: Main model class combining GAT and Transformer layers
  - `TransformerFusion`: Cross-modal attention between RNA and ADT
  - `SparseCrossAttentionLayer`: Efficient sparse attention mechanism
  - `GraphPositionalEncoding`: Graph-aware positional encoding
  - `AdapterLayer`: Parameter-efficient fine-tuning adapters

### Data Processing (`scripts/data_provider/`)

- **`data_preprocessing.py`** - Data normalization and preparation
  - `clr_normalize()`: Centered Log-Ratio normalization for ADT data
  - `zscore_normalize()`: Z-score normalization
  - `prepare_train_test_anndata()`: Train/test split and preprocessing pipeline

- **`graph_data_builder.py`** - Graph construction from single-cell data
  - `build_pyg_data()`: Converts AnnData to PyTorch Geometric format
  - `sparsify_graph()`: Reduces graph density for memory efficiency
  - `setup_graph_processing()`: Configures graph parameters based on GPU memory

### Training (`scripts/trainer/`)

- **`gat_trainer.py`** - Main training pipeline
  - `train_gat_transformer_fusion()`: Complete training workflow
  - Handles multi-task learning (ADT regression + AML classification)
  - Supports cell type classification
  - Mixed precision training and early stopping

- **`fineTune.py`** - Transfer learning and fine-tuning
  - `load_and_finetune()`: Load pre-trained model and fine-tune on new data
  - Handles different ADT marker counts
  - Supports freezing/unfreezing encoder layers

### Visualization (`scripts/`)

- **`visualizations.py`** - Plotting functions
  - UMAP visualizations
  - ADT marker correlation plots
  - Gene-protein relationship networks
  - Attention weight visualizations

## 📂 Tutorials/ - Jupyter Notebooks

- **`Training.ipynb`** - Complete training workflow example
- **`Test.ipynb`** - Model evaluation and testing
- **`Finetune.ipynb`** - Fine-tuning pre-trained models on new datasets
- **`scVI.ipynb`** - Integration with scVI (if applicable)

## 📂 R/ - R Scripts (Legacy/Alternative Processing)

- **`DataProcessing.R`** - R-based data preprocessing pipeline
- **`AMLTirated.R`** - AML-specific data processing
- **`GSM6805326.R`** - Dataset-specific processing
- **`RefMapping.R`** - Reference mapping utilities
- **`WNN_Mapping.R`** - Weighted Nearest Neighbor mapping

## 📂 publication_figures/ - Output Visualizations

Contains generated figures and plots from analysis pipelines.

## 🗂️ Key Data Flow

1. **Data Input** → `data_preprocessing.py` → Normalized AnnData
2. **Graph Construction** → `graph_data_builder.py` → PyTorch Geometric Data
3. **Model Definition** → `doNET.py` → `GATWithTransformerFusion`
4. **Training** → `gat_trainer.py` → Trained model weights
5. **Fine-tuning** → `fineTune.py` → Adapted model for new data
6. **Visualization** → `visualizations.py` → Analysis plots

## 🔑 Main Entry Points

- **Training**: `scripts/trainer/gat_trainer.py::train_gat_transformer_fusion()`
- **Fine-tuning**: `scripts/trainer/fineTune.py::load_and_finetune()`
- **Model**: `scripts/model/doNET.py::GATWithTransformerFusion`
- **Data Prep**: `scripts/data_provider/data_preprocessing.py::prepare_train_test_anndata()`

