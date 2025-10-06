"""
GSM3587990.py - Combine GSE116256 dataset files into AnnData object

This script processes the GSE116256 dataset files and combines them into a single AnnData object
for use with the DeepOMAPNet model testing pipeline.

Dataset: GSE116256 - Single-cell RNA-seq of AML samples
Files: GSM3587923-GSM3588005 (dem.txt.gz and anno.txt.gz files)
"""

import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse
import gzip
import warnings
warnings.filterwarnings('ignore')

def read_dem_file(filepath):
    """
    Read a .dem.txt.gz file (expression matrix)
    """
    print(f"Reading expression file: {os.path.basename(filepath)}")

    with gzip.open(filepath, 'rt') as f:
        first_line = f.readline().strip()
        n_genes, n_cells = map(int, first_line.split())

        f.seek(0)
        data = []
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                data.append([int(parts[0])-1, int(parts[1])-1, float(parts[2])])

    if data:
        data = np.array(data)
        row_indices = data[:, 0].astype(int)
        col_indices = data[:, 1].astype(int)
        values = data[:, 2]

        X = sparse.csr_matrix((values, (row_indices, col_indices)), shape=(n_genes, n_cells))
    else:
        X = sparse.csr_matrix((n_genes, n_cells))

    return X, n_genes, n_cells

def read_anno_file(filepath):
    """
    Read a .anno.txt.gz file (cell annotations)
    """
    print(f"Reading annotation file: {os.path.basename(filepath)}")

    with gzip.open(filepath, 'rt') as f:
        header = f.readline().strip().split('\t')

        data = []
        for line in f:
            data.append(line.strip().split('\t'))

    df = pd.DataFrame(data, columns=header)

    return df

def extract_sample_info(filename):
    """
    Extract sample information from filename
    """
    base_name = filename.replace('.dem.txt.gz', '').replace('.anno.txt.gz', '')

    parts = base_name.split('_')
    gsm_id = parts[0]

    if len(parts) > 1:
        sample_info = parts[1]

        if sample_info.startswith('AML'):
            if '-' in sample_info:
                sample_id, timepoint = sample_info.split('-', 1)
                sample_type = 'AML'
            else:
                sample_id = sample_info
                timepoint = 'D0'
                sample_type = 'AML'
        elif sample_info.startswith('BM'):
            sample_id = sample_info
            timepoint = 'D0'
            sample_type = 'BM'
        elif sample_info in ['MUTZ3', 'OCI-AML3']:
            sample_id = sample_info
            timepoint = 'D0'
            sample_type = 'CellLine'
        else:
            sample_id = sample_info
            timepoint = 'D0'
            sample_type = 'Unknown'
    else:
        sample_id = gsm_id
        timepoint = 'D0'
        sample_type = 'Unknown'

    return {
        'gsm_id': gsm_id,
        'sample_id': sample_id,
        'timepoint': timepoint,
        'sample_type': sample_type
    }

def process_gsm_files(data_dir):
    """
    Process all GSM files and combine into AnnData object
    """
    print("Processing GSE116256 dataset files...")

    all_files = os.listdir(data_dir)
    dem_files = [f for f in all_files if f.endswith('.dem.txt.gz')]
    anno_files = [f for f in all_files if f.endswith('.anno.txt.gz')]

    print(f"Found {len(dem_files)} expression files and {len(anno_files)} annotation files")

    dem_files.sort()
    anno_files.sort()

    expression_matrices = []
    cell_annotations = []
    sample_info_list = []
    gene_names = None

    for i, dem_file in enumerate(dem_files):
        anno_file = dem_file.replace('.dem.txt.gz', '.anno.txt.gz')

        if anno_file not in anno_files:
            print(f"Warning: No annotation file found for {dem_file}")
            continue

        sample_info = extract_sample_info(dem_file)
        sample_info['file_index'] = i

        dem_path = os.path.join(data_dir, dem_file)
        X, n_genes, n_cells = read_dem_file(dem_path)

        anno_path = os.path.join(data_dir, anno_file)
        anno_df = read_anno_file(anno_path)

        if len(anno_df) != n_cells:
            print(f"Warning: Annotation file {anno_file} has {len(anno_df)} cells, but expression has {n_cells}")
            if len(anno_df) > n_cells:
                anno_df = anno_df.iloc[:n_cells]
            else:
                padding = pd.DataFrame(index=range(n_cells - len(anno_df)))
                for col in anno_df.columns:
                    padding[col] = 'Unknown'
                anno_df = pd.concat([anno_df, padding], ignore_index=True)

        anno_df['gsm_id'] = sample_info['gsm_id']
        anno_df['sample_id'] = sample_info['sample_id']
        anno_df['timepoint'] = sample_info['timepoint']
        anno_df['sample_type'] = sample_info['sample_type']
        anno_df['file_index'] = i

        expression_matrices.append(X)
        cell_annotations.append(anno_df)
        sample_info_list.append(sample_info)

        print(f"Processed {dem_file}: {n_cells} cells, {n_genes} genes")

    print("\nCombining expression matrices...")
    combined_X = sparse.hstack(expression_matrices, format='csr')

    print("Combining cell annotations...")
    combined_obs = pd.concat(cell_annotations, ignore_index=True)

    combined_obs['cell_id'] = [f"{row['gsm_id']}_{i}" for i, row in combined_obs.iterrows()]
    combined_obs.index = combined_obs['cell_id']

    n_genes_total = combined_X.shape[0]
    gene_names = [f"Gene_{i+1:05d}" for i in range(n_genes_total)]

    var_df = pd.DataFrame(index=gene_names)
    var_df['gene_id'] = gene_names
    var_df['feature_type'] = 'Gene Expression'

    print("Creating AnnData object...")
    adata = ad.AnnData(
        X=combined_X.T,
        obs=combined_obs,
        var=var_df
    )

    adata.obs['n_genes'] = (adata.X > 0).sum(axis=1).A1
    adata.obs['total_counts'] = adata.X.sum(axis=1).A1

    adata.uns['sample_info'] = sample_info_list
    adata.uns['dataset'] = 'GSE116256'
    adata.uns['description'] = 'Single-cell RNA-seq of AML samples'

    print(f"\n✅ Successfully created AnnData object:")
    print(f"   • Total cells: {adata.n_obs:,}")
    print(f"   • Total genes: {adata.n_vars:,}")
    print(f"   • Samples: {len(sample_info_list)}")
    print(f"   • Memory usage: {adata.X.data.nbytes / 1024**2:.1f} MB")

    return adata

def save_ann_data(adata, output_path):
    """
    Save AnnData object to file
    """
    print(f"\nSaving AnnData object to {output_path}...")
    adata.write(output_path)
    print("✅ AnnData object saved successfully!")

def main():
    """
    Main function to process GSE116256 dataset
    """
    data_dir = "/projects/vanaja_lab/satya/Datasets/GSE116256"
    output_path = "GSE116256_combined.h5ad"

    print("="*60)
    print("GSE116256 Dataset Processing")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output file: {output_path}")
    print()

    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory {data_dir} does not exist!")
        return

    try:
        adata = process_gsm_files(data_dir)

        save_ann_data(adata, output_path)

        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total cells: {adata.n_obs:,}")
        print(f"Total genes: {adata.n_vars:,}")
        print(f"Number of samples: {len(adata.uns['sample_info'])}")

        sample_type_counts = adata.obs['sample_type'].value_counts()
        print(f"\nSample type distribution:")
        for sample_type, count in sample_type_counts.items():
            print(f"  {sample_type}: {count:,} cells")

        timepoint_counts = adata.obs['timepoint'].value_counts()
        print(f"\nTimepoint distribution:")
        for timepoint, count in timepoint_counts.items():
            print(f"  {timepoint}: {count:,} cells")

        print(f"\nQuality metrics:")
        print(f"  Mean genes per cell: {adata.obs['n_genes'].mean():.1f}")
        print(f"  Mean counts per cell: {adata.obs['total_counts'].mean():.1f}")
        print(f"  Cells with >500 genes: {(adata.obs['n_genes'] > 500).sum():,}")
        print(f"  Cells with >1000 genes: {(adata.obs['n_genes'] > 1000).sum():,}")

        print(f"\n✅ Processing completed successfully!")
        print(f"   Output file: {output_path}")
        print(f"   File size: {os.path.getsize(output_path) / 1024**2:.1f} MB")

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
