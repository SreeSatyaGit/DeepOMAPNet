# GSE116256 Dataset Processing

This directory contains scripts to process the GSE116256 dataset and test it with the trained DeepOMAPNet models.

## Dataset Information

- **Dataset**: GSE116256 - Single-cell RNA-seq of AML samples
- **Files**: GSM3587923-GSM3588005 (dem.txt.gz and anno.txt.gz files)
- **Location**: `/projects/vanaja_lab/satya/Datasets/GSE116256`
- **Description**: Single-cell RNA sequencing data from AML (Acute Myeloid Leukemia) samples

## Files

### Processing Scripts

1. **`GSM3587990.py`** - Complete processing script with detailed error handling
2. **`load_gse116256.py`** - Simplified loader for easy integration
3. **`run_gse116256_processing.py`** - Runner script to execute processing

### Testing

4. **`Notebooks/Test_New_Dataset.ipynb`** - Jupyter notebook for testing trained models

## Usage

### Option 1: Quick Processing

```bash
# Run the processing script
python run_gse116256_processing.py
```

### Option 2: Manual Processing

```python
# In Python
from load_gse116256 import load_gse116256_dataset

# Load the dataset
adata = load_gse116256_dataset(
    data_dir="/projects/vanaja_lab/satya/Datasets/GSE116256",
    output_file="GSE116256_combined.h5ad",
    force_reload=False
)
```

### Option 3: Detailed Processing

```bash
# Run the complete processing script
python GSM3587990.py
```

## Testing with Trained Models

1. **Process the dataset** (using any option above)
2. **Open the testing notebook**:
   ```bash
   jupyter notebook Notebooks/Test_New_Dataset.ipynb
   ```
3. **Uncomment the GSE116256 loading section** in cell 2
4. **Run the notebook** to test the trained models

## Output

The processing creates:
- **`GSE116256_combined.h5ad`** - Combined AnnData object with all samples
- **Metadata** including sample IDs, GSM IDs, and quality metrics

## Dataset Structure

The processed dataset contains:
- **Cells**: All cells from all samples combined
- **Genes**: Common gene set across all samples
- **Metadata**: 
  - `sample_id`: Original sample identifier
  - `gsm_id`: GSM identifier
  - `n_genes`: Number of genes expressed per cell
  - `total_counts`: Total UMI counts per cell

## Sample Types

The dataset includes:
- **AML samples**: Various AML patient samples at different timepoints
- **BM samples**: Bone marrow samples
- **Cell lines**: MUTZ3, OCI-AML3 cell lines

## Memory Requirements

- **Raw files**: ~2-5 GB
- **Processed dataset**: ~1-3 GB (depending on sparsity)
- **RAM usage**: ~4-8 GB during processing

## Troubleshooting

### Common Issues

1. **Memory errors**: Process in smaller batches by modifying `batch_size` in `load_gse116256.py`
2. **File not found**: Check that the data directory path is correct
3. **Permission errors**: Ensure write permissions in the current directory

### Performance Tips

1. **Use the simplified loader** (`load_gse116256.py`) for faster processing
2. **Set `force_reload=False`** to avoid reprocessing if the combined file exists
3. **Process in batches** for very large datasets

## Integration with DeepOMAPNet

The processed dataset can be used with:
- **RNA-to-ADT mapping**: Predict ADT embeddings from RNA data
- **Cell type annotation**: Use predicted embeddings for cell type identification
- **Cross-modal analysis**: Compare RNA and predicted ADT patterns

## Example Usage in Notebook

```python
# In Test_New_Dataset.ipynb, uncomment this section:
import sys
sys.path.append('..')
from load_gse116256 import load_gse116256_dataset

print("Loading GSE116256 dataset...")
new_rna_data = load_gse116256_dataset(
    data_dir="/projects/vanaja_lab/satya/Datasets/GSE116256",
    output_file="GSE116256_combined.h5ad",
    force_reload=False
)
new_adt_data = None  # No ADT data available
```

This will load the real GSE116256 dataset for testing with the trained models.
