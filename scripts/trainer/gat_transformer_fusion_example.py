"""
Example usage of GATWithTransformerFusion training
"""

import torch
import numpy as np
from scripts.data_provider.graph_data_builder import build_pyg_data
from scripts.trainer.gat_trainer import train_gat_transformer_fusion
from scripts.model.doNET import GATWithTransformerFusion

def train_gat_transformer_fusion_example(rna_adata, adt_adata, epochs=50):
    """
    Example function to train GATWithTransformerFusion model
    
    Args:
        rna_adata: AnnData object with RNA data
        adt_adata: AnnData object with ADT data
        epochs: Number of training epochs
    
    Returns:
        trained_model, rna_pyg_data, adt_pyg_data
    """
    
    print("=== GATWithTransformerFusion Training Example ===")
    
    # Ensure same number of cells
    if rna_adata.n_obs != adt_adata.n_obs:
        print("Warning: RNA and ADT data have different number of cells")
        common_cells = rna_adata.obs_names.intersection(adt_adata.obs_names)
        rna_adata = rna_adata[common_cells]
        adt_adata = adt_adata[common_cells]
        print(f"Using {len(common_cells)} common cells")
    
    # Convert to PyTorch Geometric format
    print("Converting to PyTorch Geometric format...")
    rna_pyg_data = build_pyg_data(rna_adata)
    adt_pyg_data = build_pyg_data(adt_adata)
    
    print(f"RNA PyG data: {rna_pyg_data}")
    print(f"ADT PyG data: {adt_pyg_data}")
    
    # Train the model
    print("Starting training...")
    trained_model, rna_data, adt_data = train_gat_transformer_fusion(
        rna_data=rna_pyg_data,
        adt_data=adt_pyg_data,
        epochs=epochs,
        use_cpu_fallback=True,  # Set to False if you have sufficient GPU memory
        seed=42
    )
    
    print("Training completed!")
    return trained_model, rna_data, adt_data

def predict_with_gat_transformer_fusion(model, rna_adata):
    """
    Make predictions using trained GATWithTransformerFusion model
    
    Args:
        model: Trained GATWithTransformerFusion model
        rna_adata: AnnData object with RNA data
    
    Returns:
        predicted_adt_embeddings, fused_embeddings
    """
    
    print("Making predictions with GATWithTransformerFusion...")
    
    # Convert to PyTorch Geometric format
    rna_pyg_data = build_pyg_data(rna_adata)
    
    # Move to same device as model
    device = next(model.parameters()).device
    rna_pyg_data = rna_pyg_data.to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predicted_adt, fused_embeddings = model(
            x=rna_pyg_data.x,
            edge_index_rna=rna_pyg_data.edge_index
        )
    
    # Convert to numpy
    predicted_adt_np = predicted_adt.cpu().numpy()
    fused_embeddings_np = fused_embeddings.cpu().numpy()
    
    print(f"Predicted ADT shape: {predicted_adt_np.shape}")
    print(f"Fused embeddings shape: {fused_embeddings_np.shape}")
    
    return predicted_adt_np, fused_embeddings_np

