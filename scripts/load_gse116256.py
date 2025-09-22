#!/usr/bin/env python3
"""
load_gse116256.py - Simplified loader for GSE116256 dataset

This is a simplified version that can be easily integrated with the testing notebook.
"""

import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse
import gzip

def load_gse116256_dataset(data_dir="/projects/vanaja_lab/satya/Datasets/GSE116256", 
                          output_file="GSE116256_combined.h5ad",
                          force_reload=False):
    """
    Load GSE116256 dataset and return AnnData object
    
    Parameters:
    -----------
    data_dir : str
        Path to the GSE116256 dataset directory
    output_file : str
        Path to save the combined dataset
    force_reload : bool
        If True, reload from raw files even if output_file exists
        
    Returns:
    --------
    adata : AnnData
        Combined dataset
    """
    
    # Check if combined file already exists
    if os.path.exists(output_file) and not force_reload:
        print(f"Loading existing combined dataset: {output_file}")
        return sc.read_h5ad(output_file)
    
    print("Processing GSE116256 dataset from raw files...")
    
    # Get all files
    all_files = os.listdir(data_dir)
    dem_files = sorted([f for f in all_files if f.endswith('.dem.txt.gz')])
    anno_files = sorted([f for f in all_files if f.endswith('.anno.txt.gz')])
    
    print(f"Found {len(dem_files)} expression files and {len(anno_files)} annotation files")
    
    # Process files in batches to manage memory
    batch_size = 10
    all_matrices = []
    all_annotations = []
    
    for i in range(0, len(dem_files), batch_size):
        batch_dem = dem_files[i:i+batch_size]
        batch_anno = anno_files[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(dem_files)-1)//batch_size + 1}")
        
        batch_matrices = []
        batch_annotations = []
        
        for dem_file, anno_file in zip(batch_dem, batch_anno):
            try:
                # Read expression data
                dem_path = os.path.join(data_dir, dem_file)
                with gzip.open(dem_path, 'rt') as f:
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
                    X = sparse.csr_matrix(
                        (data[:, 2], (data[:, 0], data[:, 1])), 
                        shape=(n_genes, n_cells)
                    )
                else:
                    X = sparse.csr_matrix((n_genes, n_cells))
                
                # Read annotations
                anno_path = os.path.join(data_dir, anno_file)
                with gzip.open(anno_path, 'rt') as f:
                    header = f.readline().strip().split('\t')
                    anno_data = [line.strip().split('\t') for line in f]
                
                anno_df = pd.DataFrame(anno_data, columns=header)
                
                # Ensure correct number of cells
                if len(anno_df) != n_cells:
                    if len(anno_df) > n_cells:
                        anno_df = anno_df.iloc[:n_cells]
                    else:
                        padding = pd.DataFrame(index=range(n_cells - len(anno_df)))
                        for col in anno_df.columns:
                            padding[col] = 'Unknown'
                        anno_df = pd.concat([anno_df, padding], ignore_index=True)
                
                # Add sample info
                sample_id = dem_file.replace('.dem.txt.gz', '')
                anno_df['sample_id'] = sample_id
                anno_df['gsm_id'] = sample_id.split('_')[0]
                
                batch_matrices.append(X)
                batch_annotations.append(anno_df)
                
            except Exception as e:
                print(f"Warning: Error processing {dem_file}: {e}")
                continue
        
        if batch_matrices:
            # Combine batch
            batch_combined = sparse.hstack(batch_matrices, format='csr')
            batch_anno_combined = pd.concat(batch_annotations, ignore_index=True)
            
            all_matrices.append(batch_combined)
            all_annotations.append(batch_anno_combined)
    
    if not all_matrices:
        raise ValueError("No valid files could be processed")
    
    # Combine all batches
    print("Combining all batches...")
    combined_X = sparse.hstack(all_matrices, format='csr')
    combined_obs = pd.concat(all_annotations, ignore_index=True)
    
    # Create cell and gene names
    combined_obs['cell_id'] = [f"Cell_{i:06d}" for i in range(len(combined_obs))]
    combined_obs.index = combined_obs['cell_id']
    
    gene_names = [f"Gene_{i+1:05d}" for i in range(combined_X.shape[0])]
    var_df = pd.DataFrame(index=gene_names)
    var_df['gene_id'] = gene_names
    
    # Create AnnData object
    adata = ad.AnnData(
        X=combined_X.T,  # Transpose to cells x genes
        obs=combined_obs,
        var=var_df
    )
    
    # Add basic statistics
    adata.obs['n_genes'] = (adata.X > 0).sum(axis=1).A1
    adata.obs['total_counts'] = adata.X.sum(axis=1).A1
    
    # Add metadata
    adata.uns['dataset'] = 'GSE116256'
    adata.uns['description'] = 'Single-cell RNA-seq of AML samples'
    
    # Save the combined dataset
    print(f"Saving combined dataset to {output_file}...")
    adata.write(output_file)
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Cells: {adata.n_obs:,}")
    print(f"   Genes: {adata.n_vars:,}")
    print(f"   Samples: {adata.obs['sample_id'].nunique()}")
    
    return adata

# Example usage
if __name__ == "__main__":
    # Load the dataset
    adata = load_gse116256_dataset()
    
    # Display basic info
    print(f"\nDataset shape: {adata.shape}")
    print(f"Sample IDs: {adata.obs['sample_id'].unique()[:10]}...")  # Show first 10
    print(f"Memory usage: {adata.X.data.nbytes / 1024**2:.1f} MB")
