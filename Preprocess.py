import scanpy as sc
import anndata as ad
import numpy as np
from typing import Dict, Tuple
import anndata


def prepare_train_test_anndata(
    GSM_Controls_RNA=sc.read_h5ad("/projects/vanaja_lab/satya/Datasets/GSMControlRNA.h5ad"),
    GSM_Controls_ADT=sc.read_h5ad("/projects/vanaja_lab/satya/Datasets/ControlADT.h5ad"),
    GSM_AML_RNA_A=sc.read_h5ad("/projects/vanaja_lab/satya/Datasets/AMLARNA.h5ad"),
    GSM_AML_ADT_A=sc.read_h5ad("/projects/vanaja_lab/satya/Datasets/AMLAADT.h5ad"),
    GSM_AML_RNA_B=sc.read_h5ad("/projects/vanaja_lab/satya/Datasets/AMLBRNA.h5ad"),
    GSM_AML_ADT_B=sc.read_h5ad("/projects/vanaja_lab/satya/Datasets/AMLBADT.h5ad"),
):
    # Concatenate gene data
    adata_gene = anndata.concat(
        [GSM_AML_RNA_B, GSM_AML_RNA_A, GSM_Controls_RNA],
        join="outer",
        label="source",
        keys=["GSM_AML_RNA_B", "GSM_AML_RNA_A", "GSM_Controls_RNA"],
    )

    # Concatenate protein data
    adata_protein = anndata.concat(
        [GSM_Controls_ADT, GSM_AML_ADT_A, GSM_AML_ADT_B],
        join="outer",
        label="source",
        keys=["GSM_Controls_ADT", "GSM_AML_ADT_A", "GSM_AML_ADT_B"],
    )

    # Inspect sample IDs in the gene data
    print("All sample IDs in gene data:", adata_gene.obs["samples"].unique())

    samples = list(adata_gene.obs["samples"].unique())

    # Separate into AML and Control
    aml_samples = [s for s in samples if s.startswith("AML")]
    control_samples = [s for s in samples if s.startswith("Control")]

    # Shuffle for random split
    np.random.seed(42)  # for reproducibility
    aml_samples = np.random.permutation(aml_samples)
    control_samples = np.random.permutation(control_samples)

    # Calculate split indices
    def split_indices(n, frac=0.8):
        split_at = int(np.ceil(frac * n))
        return split_at

    aml_split = split_indices(len(aml_samples), 0.8)
    control_split = split_indices(len(control_samples), 0.8)

    aml_train = aml_samples[:aml_split].tolist()
    aml_test = aml_samples[aml_split:].tolist()
    control_train = control_samples[:control_split].tolist()
    control_test = control_samples[control_split:].tolist()

    print("AML 80% train:", aml_train)
    print("AML 20% test:", aml_test)
    print("Control 80% train:", control_train)
    print("Control 20% test:", control_test)

    # Define your training samples exactly as they appear
    train_samples = aml_train + control_train

    # Build a boolean mask for training cells based on 'samples'
    train_mask_gene = adata_gene.obs["samples"].isin(train_samples)
    train_mask_protein = adata_protein.obs["samples"].isin(train_samples)

    # Subset the gene and protein AnnData objects for training and test sets
    adata_gene_train = adata_gene[train_mask_gene].copy()
    adata_protein_train = adata_protein[train_mask_protein].copy()
    adata_gene_test = adata_gene[~train_mask_gene].copy()
    adata_protein_test = adata_protein[~train_mask_protein].copy()

    # For both training and test sets, align cell IDs by taking the intersection and sorting
    def align_obs(gene_data, protein_data):
        common_cells = sorted(
            set(gene_data.obs_names).intersection(set(protein_data.obs_names))
        )
        gene_data_aligned = gene_data[common_cells].copy()
        protein_data_aligned = protein_data[common_cells].copy()
        assert all(
            gene_data_aligned.obs_names == protein_data_aligned.obs_names
        ), "Cell IDs do not match after alignment!"
        return gene_data_aligned, protein_data_aligned

    adata_gene_train, adata_protein_train = align_obs(
        adata_gene_train, adata_protein_train
    )
    adata_gene_test, adata_protein_test = align_obs(
        adata_gene_test, adata_protein_test
    )

    print("Train cells:", adata_gene_train.n_obs, "| Test cells:", adata_gene_test.n_obs)

    # Convert data matrices to float32 if needed
    adata_gene_train.X = adata_gene_train.X.astype("float32")
    adata_gene_test.X = adata_gene_test.X.astype("float32")
    adata_protein_train.X = adata_protein_train.X.astype("float32")
    adata_protein_test.X = adata_protein_test.X.astype("float32")

    return (
        adata_gene_train,
        adata_gene_test,
        adata_protein_train,
        adata_protein_test,
    )
