# Scripts Directory Structure

The scripts directory has been reorganized into three logical modules for better organization and maintainability.

## Directory Structure

```
scripts/
├── __init__.py
├── data_provider/
│   ├── __init__.py
│   ├── graph_data_builder.py         # PyTorch Geometric data conversion
│   ├── data_preprocessing.py         # Data preprocessing utilities
│   ├── gse116256_loader.py          # GSE116256 dataset loader
│   ├── gse116256_pipeline.py        # GSE116256 processing pipeline
│   ├── gsm3587990_loader.py         # GSM3587990 dataset loader
│   └── README_GSE116256.md          # GSE116256 documentation
├── model/
│   ├── __init__.py
│   ├── gat_models.py                # GAT model architectures
│   └── transformer_models.py        # Transformer mapping model
└── trainer/
    ├── __init__.py
    ├── adt_predictor.py             # ADT prediction pipeline
    ├── gat_trainer.py               # GAT training utilities
    └── rna_gat_finetuner.py         # RNA GAT fine-tuning
```

## Module Descriptions

### 📊 Data Provider (`data_provider/`)
Contains all data-related functionality:
- **Data Loading**: Load various datasets (GSE116256, GSM3587990)
- **Preprocessing**: Standard preprocessing pipelines
- **Embedding Extraction**: Convert AnnData to PyTorch Geometric format
- **Processing Pipelines**: Complete data processing workflows

**Key Files:**
- `graph_data_builder.py`: Core function `build_pyg_data()` for PyG conversion
- `data_preprocessing.py`: Standard preprocessing utilities
- `gse116256_loader.py`: GSE116256 dataset loader
- `gse116256_pipeline.py`: Complete processing pipeline

### Model (`model/`)
Contains neural network architectures:
- **GAT Models**: Graph Attention Network implementations
- **Transformer Models**: RNA-to-ADT mapping transformers

**Key Files:**
- `gat_models.py`: `SimpleGAT` and `GAT` classes
- `transformer_models.py`: `TransformerMapping` class

###  Trainer (`trainer/`)
Contains training and prediction functionality:
- **Model Training**: GAT model training utilities
- **Fine-tuning**: RNA GAT fine-tuning on new datasets
- **Predictions**: ADT prediction pipeline

**Key Files:**
- `gat_trainer.py`: `train_gat_model()` function
- `rna_gat_finetuner.py`: `RNAGATFineTuner` class and fine-tuning methods
- `adt_predictor.py`: `ADTPredictor` class and prediction pipeline

## Usage Examples

### Importing from the new structure:

```python
# Import from specific modules
from scripts.data_provider import build_pyg_data
from scripts.model import SimpleGAT, TransformerMapping
from scripts.trainer import ADTPredictor, RNAGATFineTuner

# Or import everything
from scripts import data_provider, model, trainer
```

### Using the prediction pipeline:

```python
from scripts.trainer import ADTPredictor

predictor = ADTPredictor(
    individual_models_dir="path/to/trained_models/individual_models_YYYYMMDD_HHMMSS"
)

# Make predictions
adata_with_predictions = predictor.add_predictions_to_adata(
    adata=your_rna_data,
    predict_cell_types=True
)
```

### Fine-tuning models:

```python
from scripts.trainer import fine_tune_rna_gat

fine_tuned_model = fine_tune_rna_gat(
    adata=your_new_data,
    individual_models_dir="path/to/pretrained_models",
    method='unsupervised',
    epochs=100
)
```



