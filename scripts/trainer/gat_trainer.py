"""
GAT Transformer Fusion Training Module

This module provides the main training pipeline for DeepOMAPNet,
a Graph Attention Network with Transformer Fusion for RNA-to-ADT mapping
in single-cell CITE-seq data.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional, Union
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr


def compute_multi_task_loss(adt_pred, adt_target, aml_pred, aml_target, 
                           adt_weight=1.0, classification_weight=1.0):
    """
    Compute combined loss for both ADT prediction and AML classification tasks
    
    Args:
        adt_pred: ADT predictions [N, num_markers]
        adt_target: ADT targets [N, num_markers] 
        aml_pred: AML predictions [N, 1]
        aml_target: AML labels [N] (0=Normal, 1=AML)
        adt_weight: Weight for ADT regression loss
        classification_weight: Weight for classification loss
    
    Returns:
        total_loss: Combined weighted loss
        adt_loss: ADT regression loss
        aml_loss: AML classification loss
    """
    # ADT regression loss (MSE)
    adt_loss = F.mse_loss(adt_pred, adt_target)
    
    # AML classification loss (Binary Cross-Entropy with logits)
    aml_loss = F.binary_cross_entropy_with_logits(
        aml_pred.squeeze(), aml_target.float()
    )
    
    # Combined loss
    total_loss = adt_weight * adt_loss + classification_weight * aml_loss
    
    return total_loss, adt_loss, aml_loss


def compute_classification_metrics(aml_pred, aml_target):
    """
    Compute classification metrics for AML prediction
    
    Args:
        aml_pred: AML predictions [N, 1] (logits)
        aml_target: AML labels [N] (0=Normal, 1=AML)
    
    Returns:
        metrics: Dictionary of classification metrics
    """
    # Convert logits to probabilities
    aml_probs = torch.sigmoid(aml_pred).cpu().numpy().squeeze()
    aml_pred_binary = (aml_probs > 0.5).astype(int)
    aml_target_np = aml_target.cpu().numpy()
    
    # Compute metrics
    accuracy = accuracy_score(aml_target_np, aml_pred_binary)
    precision = precision_score(aml_target_np, aml_pred_binary, zero_division=0)
    recall = recall_score(aml_target_np, aml_pred_binary, zero_division=0)
    f1 = f1_score(aml_target_np, aml_pred_binary, zero_division=0)
    
    # AUC-ROC (handle case where all labels are the same)
    try:
        auc_roc = roc_auc_score(aml_target_np, aml_probs)
    except ValueError:
        auc_roc = 0.5  # Random classifier performance
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc
    }


def train_gat_transformer_fusion(
    rna_data,
    adt_data,
    aml_labels=None,  # AML classification labels (0=Normal, 1=AML)
    rna_anndata=None,  # AnnData object for RNA preprocessing
    adt_anndata=None,  # AnnData object for ADT preprocessing
    epochs: int = 200,
    use_cpu_fallback: bool = False,
    seed: int = 42,
    stratify_labels: Optional[np.ndarray] = None,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    adt_weight: float = 1.0,  # Weight for ADT regression loss
    classification_weight: float = 1.0,  # Weight for AML classification loss
    dropout_rate: float = 0.4,
    hidden_channels: int = 96,
    num_heads: int = 8,
    num_attention_heads: int = 8,
    num_layers: int = 3,
    use_mixed_precision: bool = True,
    early_stopping_patience: int = 20,
) -> Tuple[torch.nn.Module, object, object, Dict]:
    """
    Train GAT Transformer Fusion model for RNA-to-ADT mapping.
    
    Args:
        rna_data: PyTorch Geometric data object containing RNA features and graph structure
        adt_data: PyTorch Geometric data object containing ADT targets
        rna_anndata: AnnData object for RNA preprocessing (cells x genes). 
                     If None, will use rna_data.x directly without preprocessing.
        adt_anndata: AnnData object for ADT preprocessing (CLR normalization, etc.). 
                     If None, will use adt_data.x directly without preprocessing.
        epochs: Number of training epochs
        use_cpu_fallback: Whether to fallback to CPU if GPU OOM occurs
        seed: Random seed for reproducibility
        stratify_labels: Labels for stratified train/val/test splitting
        train_fraction: Fraction of data to use for training
        val_fraction: Fraction of data to use for validation
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        dropout_rate: Dropout rate for model layers
        hidden_channels: Number of hidden channels in the model
        num_heads: Number of attention heads in GAT layers
        num_attention_heads: Number of attention heads in transformer layers
        num_layers: Number of transformer fusion layers
        use_mixed_precision: Whether to use automatic mixed precision
        early_stopping_patience: Number of epochs to wait before early stopping
        
    Returns:
        Tuple containing:
        - trained_model: The trained GAT Transformer Fusion model
        - rna_data: RNA data with train/val/test masks
        - adt_data: ADT data (preprocessed)
        - training_history: Dictionary containing training metrics per epoch
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # Set memory fraction to prevent OOM - more aggressive
        torch.cuda.set_per_process_memory_fraction(0.5)
        print(f"GPU memory fraction set to 50%")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu_fallback else 'cpu')
    print(f"Using device: {device}")

    # Validate data dimensions
    rna_input_dim = rna_anndata.shape[1] if rna_anndata is not None else rna_data.x.size(1)
    adt_output_dim = adt_anndata.shape[1] if adt_anndata is not None else adt_data.x.size(1)
    num_nodes = rna_data.num_nodes
    
    assert adt_data.num_nodes == num_nodes, "RNA and ADT must have same number of nodes (aligned cells)."

    # Create train/validation/test splits
    train_mask, val_mask, test_mask = _create_data_splits(
        num_nodes, stratify_labels, train_fraction, val_fraction, seed
    )
    
    # Store masks in data objects
    rna_data.train_mask = train_mask
    rna_data.val_mask = val_mask
    rna_data.test_mask = test_mask
    
    print(f"Data splits — train: {int(train_mask.sum())}, "
          f"val: {int(val_mask.sum())}, test: {int(test_mask.sum())}")

    # Preprocess RNA data if AnnData provided
    if rna_anndata is not None:
        rna_input_dim = _preprocess_rna_data(rna_data, rna_anndata)
        print(f"Updated RNA input dimension after preprocessing: {rna_input_dim}")

    # Preprocess ADT data
    adt_mean, adt_std = _preprocess_adt_data(adt_data, adt_anndata)
    
    # Process AML labels efficiently and split according to data splits
    if aml_labels is not None:
        # Convert to numpy first to avoid GPU memory issues during processing
        if isinstance(aml_labels, torch.Tensor):
            aml_labels_np = aml_labels.cpu().numpy()
        else:
            aml_labels_np = np.array(aml_labels)
        
        # Count labels efficiently on CPU
        normal_count = np.sum(aml_labels_np == 0)
        aml_count = np.sum(aml_labels_np == 1)
        
        print(f"AML labels processed: {aml_labels_np.shape}, "
              f"Normal: {normal_count}, AML: {aml_count}")
        
        # Split AML labels according to the same train/val/test splits
        aml_labels_train = aml_labels_np[train_mask.cpu().numpy()]
        aml_labels_val = aml_labels_np[val_mask.cpu().numpy()]
        aml_labels_test = aml_labels_np[test_mask.cpu().numpy()]
        
        # Count labels in each split
        train_normal = np.sum(aml_labels_train == 0)
        train_aml = np.sum(aml_labels_train == 1)
        val_normal = np.sum(aml_labels_val == 0)
        val_aml = np.sum(aml_labels_val == 1)
        test_normal = np.sum(aml_labels_test == 0)
        test_aml = np.sum(aml_labels_test == 1)
        
        print(f"AML labels split — train: {len(aml_labels_train)} (Normal: {train_normal}, AML: {train_aml}), "
              f"val: {len(aml_labels_val)} (Normal: {val_normal}, AML: {val_aml}), "
              f"test: {len(aml_labels_test)} (Normal: {test_normal}, AML: {test_aml})")
        
        # Keep as numpy array for now, will convert to tensor later
        aml_labels = aml_labels_np
    else:
        print("⚠️ No AML labels provided - will train ADT prediction only")
    
    # Update output dimension after preprocessing
    adt_output_dim = adt_data.x.size(1)
    print(f"Updated ADT output dimension after preprocessing: {adt_output_dim}")

    # Initialize model with reduced dimensions for GPU memory
    model = _initialize_model(
        rna_input_dim, adt_output_dim, 
        hidden_channels=min(hidden_channels, 32),  # Cap at 32 for GPU memory
        num_heads=min(num_heads, 2),  # Cap at 2
        num_attention_heads=min(num_attention_heads, 2),  # Cap at 2
        num_layers=min(num_layers, 1),  # Cap at 1
        dropout_rate=dropout_rate, device=device
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Clear memory before moving data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Move data to device with memory management
    rna_data, adt_data, device = _move_data_to_device(rna_data, adt_data, model, device)
    
    # Move normalization tensors to device
    adt_mean = adt_mean.to(device)
    adt_std = adt_std.to(device)
    
    # Convert AML labels to tensor and move to device efficiently
    if aml_labels is not None:
        # Convert to tensor on CPU first, then move to device
        aml_labels = torch.tensor(aml_labels, dtype=torch.float32, device=device)
        print(f"AML labels moved to {device}: {aml_labels.shape}")

    # Compute graph statistics for positional encoding
    node_degrees_rna, clustering_coeffs_rna = _compute_graph_statistics(rna_data.edge_index, num_nodes)
    node_degrees_adt, clustering_coeffs_adt = _compute_graph_statistics(
        adt_data.edge_index if hasattr(adt_data, 'edge_index') else rna_data.edge_index, 
        num_nodes
    )

    # Setup training components
    optimizer, scheduler, criterion, scaler = _setup_training_components(
        model, learning_rate, weight_decay, use_mixed_precision, device
    )

    # Training loop
    training_history = _run_training_loop(
        model, rna_data, adt_data, aml_labels, optimizer, scheduler, criterion, scaler,
        node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt,
        adt_mean, adt_std, adt_weight, classification_weight, epochs, early_stopping_patience, use_mixed_precision, device
    )

    # Final evaluation
    _print_final_metrics(model, rna_data, adt_data, aml_labels, adt_mean, adt_std, 
                        node_degrees_rna, node_degrees_adt, 
                        clustering_coeffs_rna, clustering_coeffs_adt,
                        use_mixed_precision, device)

    return model, rna_data, adt_data, training_history,adt_mean, adt_std, node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt


def _create_data_splits(
    num_nodes: int, 
    stratify_labels: Optional[np.ndarray], 
    train_fraction: float, 
    val_fraction: float, 
    seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create train/validation/test data splits."""
    indices = np.arange(num_nodes)
    rng = np.random.default_rng(seed)
    
    if stratify_labels is not None:
        # Stratified splitting
        from sklearn.model_selection import StratifiedShuffleSplit
        
        stratify_labels = np.asarray(stratify_labels)
        assert stratify_labels.shape[0] == num_nodes
        
        # Two-stage stratified split
        sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
        train_idx, rest_idx = next(sss1.split(indices, stratify_labels))
        
        rest_labels = stratify_labels[rest_idx]
        val_size = int(val_fraction * num_nodes)
        
        sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_size/len(rest_idx), random_state=seed)
        val_rel, test_rel = next(sss2.split(rest_idx, rest_labels))
        
        val_idx = rest_idx[val_rel]
        test_idx = rest_idx[test_rel]
    else:
        # Random splitting
        rng.shuffle(indices)
        n_train = int(train_fraction * num_nodes)
        n_val = int(val_fraction * num_nodes)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

    # Create boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    return train_mask, val_mask, test_mask


def _initialize_model(
    rna_input_dim: int,
    adt_output_dim: int,
    hidden_channels: int,
    num_heads: int,
    num_attention_heads: int,
    num_layers: int,
    dropout_rate: float,
    device: torch.device
) -> torch.nn.Module:
    """Initialize the GAT Transformer Fusion model."""
    from model.doNET import EnhancedGATWithTransformerFusion
    
    model = EnhancedGATWithTransformerFusion(
        in_channels=rna_input_dim,
        hidden_channels=hidden_channels,
        out_channels=adt_output_dim,
        heads=num_heads,
        dropout=dropout_rate,
        nhead=num_attention_heads,
        num_layers=num_layers,
        use_adapters=True,
        reduction_factor=4,
        adapter_l2_reg=5e-5,
        use_positional_encoding=True
    ).to(device)
    
    return model


def _preprocess_rna_data(rna_data, rna_anndata) -> int:
    """
    Preprocess RNA data using AnnData object.
    
    Args:
        rna_data: PyTorch Geometric data object
        rna_anndata: AnnData object for RNA preprocessing (cells x genes)
        
    Returns:
        Updated RNA input dimension
    """
    print("Preprocessing RNA data using AnnData...")
    
    # Convert AnnData to tensor
    if hasattr(rna_anndata.X, 'toarray'):
        rna_tensor = torch.tensor(rna_anndata.X.toarray(), dtype=torch.float32)
    else:
        rna_tensor = torch.tensor(rna_anndata.X, dtype=torch.float32)
    
    # Update PyG data
    rna_data.x = rna_tensor
    print(f"RNA preprocessing applied: {rna_data.x.shape}")
    
    return rna_data.x.size(1)


def _preprocess_adt_data(adt_data, adt_anndata=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply CLR transformation and z-score normalization to ADT data.
    
    Args:
        adt_data: PyTorch Geometric data object
        adt_anndata: AnnData object for preprocessing. If None, uses adt_data.x directly.
        
    Returns:
        Tuple of (mean, std) tensors for denormalization
    """
    print("Applying ADT preprocessing (CLR + z-score)...")
    
    if adt_anndata is not None:
        # Use AnnData for preprocessing
        print("Using AnnData object for ADT preprocessing...")
        
        # CLR transformation on AnnData
        def clr_transform_anndata(adata):
            """Apply CLR transformation to AnnData object."""
            import scanpy as sc
            from scipy import sparse
            
            # Convert to dense if sparse
            if sparse.issparse(adata.X):
                X_dense = adata.X.toarray()
            else:
                X_dense = adata.X.copy()
            
            # Add pseudocount and log transform
            X_dense = X_dense + 1.0
            X_dense = np.log(X_dense)
            
            # Center by row mean (per cell)
            row_means = np.mean(X_dense, axis=1, keepdims=True)
            X_dense = X_dense - row_means
            
            # Update AnnData
            adata.X = X_dense
            return adata
        
        # Apply CLR transformation
        adt_anndata = clr_transform_anndata(adt_anndata.copy())
        print(f"CLR transformation applied to AnnData: {adt_anndata.shape}")
        
        # Convert to tensor (now guaranteed to be dense)
        adt_tensor = torch.tensor(adt_anndata.X, dtype=torch.float32)
        
        # Update PyG data
        adt_data.x = adt_tensor
        
    else:
        # Use PyTorch Geometric data directly
        print("Using PyTorch Geometric data directly for ADT preprocessing...")
        
        # CLR transformation
        def clr_transform(x: torch.Tensor) -> torch.Tensor:
            """Apply centered log-ratio transformation per cell."""
            x_pseudo = x + 1.0
            log_x = torch.log(x_pseudo)
            geometric_means = torch.exp(log_x.mean(dim=1, keepdim=True))
            x_clr = torch.log(x_pseudo / geometric_means)
            return x_clr
        
        adt_data.x = clr_transform(adt_data.x)
        print(f"CLR transformation applied to PyG data: {adt_data.x.shape}")
    
    # Z-score normalization per marker
    adt_mean = adt_data.x.mean(dim=0, keepdim=True)
    adt_std = adt_data.x.std(dim=0, keepdim=True) + 1e-8
    adt_data.x = (adt_data.x - adt_mean) / adt_std
    
    print(f"Z-score normalization applied - mean: {adt_mean.mean().item():.4f}, "
          f"std: {adt_std.mean().item():.4f}")
    
    return adt_mean, adt_std


def _move_data_to_device(rna_data, adt_data, model, device):
    """Move data to the specified device with aggressive GPU memory management."""
    if device.type == "cuda":
        # Clear memory before moving data
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Try to move data in smaller chunks to avoid OOM
        try:
            # Move model first
            model = model.to(device)
            print(f"Model moved to {device}")
            
            # Move RNA data
            rna_data = rna_data.to(device)
            print(f"RNA data moved to {device}")
            
            # Clear cache before moving ADT data
            torch.cuda.empty_cache()
            
            # Move ADT data
            adt_data = adt_data.to(device)
            print(f"ADT data moved to {device}")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("⚠️ GPU OOM during data movement, trying memory optimization...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Try with even more aggressive memory management
                torch.cuda.set_per_process_memory_fraction(0.3)  # Reduce to 30%
                print("🔄 Reduced GPU memory fraction to 30%")
                
                try:
                    # Retry with reduced memory
                    model = model.to(device)
                    rna_data = rna_data.to(device)
                    adt_data = adt_data.to(device)
                    print(f"✅ Data successfully moved to {device} with reduced memory")
                except RuntimeError as e2:
                    if "out of memory" in str(e2).lower():
                        print("❌ Still OOM, falling back to CPU.")
                        device = torch.device('cpu')
                        model = model.cpu()
                        rna_data = rna_data.cpu()
                        adt_data = adt_data.cpu()
                    else:
                        raise e2
            else:
                raise e
    else:
        # CPU device - no special handling needed
        rna_data = rna_data.to(device)
        adt_data = adt_data.to(device)
        print(f"Data moved to {device}")
    
    return rna_data, adt_data, device


def _compute_graph_statistics(edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute graph statistics for positional encoding."""
    from model.doNET import compute_graph_statistics_fast
    return compute_graph_statistics_fast(edge_index, num_nodes)


def _setup_training_components(model, learning_rate, weight_decay, use_mixed_precision, device):
    """Setup optimizer, scheduler, loss function, and scaler."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_mixed_precision and device.type == "cuda"))
    
    return optimizer, scheduler, criterion, scaler


def _run_training_loop(
    model, rna_data, adt_data, aml_labels, optimizer, scheduler, criterion, scaler,
    node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt,
    adt_mean, adt_std, adt_weight, classification_weight, epochs, early_stopping_patience, use_mixed_precision, device
) -> Dict:
    """Run the main training loop."""
    print("Starting training...")
    
    best_val_r2 = float('-inf')
    best_state = None
    bad_epochs = 0
    
    # Global flag to disable mixed precision after first error
    mixed_precision_disabled = False
    
    training_history = {
        "epoch": [], "train_loss": [], "reg_loss": [], "aml_loss": [],
        "val_MSE": [], "val_R2": [], "test_MSE": [], "test_R2": [],
        "val_AML_Accuracy": [], "val_AML_F1": [], "test_AML_Accuracy": [], "test_AML_F1": []
    }

    for epoch in range(1, epochs + 1):
        # Training step - use global mixed precision flag
        current_mixed_precision = use_mixed_precision and not mixed_precision_disabled
        adt_loss, reg_loss, aml_loss = _training_step(
            model, rna_data, adt_data, aml_labels, optimizer, criterion, scaler,
            node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt,
            adt_mean, adt_std, adt_weight, classification_weight, epoch, epochs, current_mixed_precision, device
        )
        
        # Check if mixed precision was disabled during this step
        if not current_mixed_precision and use_mixed_precision:
            mixed_precision_disabled = True
            if epoch == 1:  # Only print once at the beginning
                print("ℹ️ Mixed precision disabled due to compatibility issues - using full precision")
        
        # Evaluation and logging
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            torch.cuda.empty_cache()
            val_metrics = _evaluate_model(
                model, rna_data, adt_data, aml_labels, adt_mean, adt_std,
                node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt,
                rna_data.val_mask, current_mixed_precision, device
            )
            test_metrics = _evaluate_model(
                model, rna_data, adt_data, aml_labels, adt_mean, adt_std,
                node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt,
                rna_data.test_mask, current_mixed_precision, device
            )

            # Update history
            training_history["epoch"].append(epoch)
            training_history["train_loss"].append(adt_loss)
            training_history["reg_loss"].append(reg_loss)
            training_history["aml_loss"].append(aml_loss)
            training_history["val_MSE"].append(val_metrics["MSE"])
            training_history["val_R2"].append(val_metrics["R2"])
            training_history["test_MSE"].append(test_metrics["MSE"])
            training_history["test_R2"].append(test_metrics["R2"])
            training_history["val_AML_Accuracy"].append(val_metrics["AML_Accuracy"])
            training_history["val_AML_F1"].append(val_metrics["AML_F1"])
            training_history["test_AML_Accuracy"].append(test_metrics["AML_Accuracy"])
            training_history["test_AML_F1"].append(test_metrics["AML_F1"])

            # Print progress
            print(f"Epoch {epoch:03d} | "
                  f"ADT Loss {adt_loss:.6f} AML Loss {aml_loss:.6f} Reg Loss {reg_loss:.6f} | "
                  f"Val MSE {val_metrics['MSE']:.6f} R² {val_metrics['R2']:.4f} | "
                  f"Test MSE {test_metrics['MSE']:.6f} R² {test_metrics['R2']:.4f} | "
                  f"Val AML Acc {val_metrics['AML_Accuracy']:.3f} F1 {val_metrics['AML_F1']:.3f} | "
                  f"Test AML Acc {test_metrics['AML_Accuracy']:.3f} F1 {test_metrics['AML_F1']:.3f}")

            # Update scheduler and check early stopping
            scheduler.step(val_metrics["MSE"])
            
            if val_metrics["R2"] > best_val_r2:
                best_val_r2 = val_metrics["R2"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no val R² improvement for {early_stopping_patience} checks)")
                    break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model (val R²={best_val_r2:.4f}).")
    
    return training_history


def _training_step(
    model, rna_data, adt_data, aml_labels, optimizer, criterion, scaler,
    node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt,
    adt_mean, adt_std, adt_weight, classification_weight, epoch, epochs, use_mixed_precision, device
) -> Tuple[float, float, float]:
    """Perform one training step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    # Clear GPU cache before training step
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    try:
        with torch.cuda.amp.autocast(enabled=(use_mixed_precision and device.type == "cuda")):
            # Forward pass
            adt_pred, aml_pred, _ = model(
                x=rna_data.x,
                edge_index_rna=rna_data.edge_index,
                edge_index_adt=adt_data.edge_index if hasattr(adt_data, 'edge_index') else None,
                node_degrees_rna=node_degrees_rna,
                node_degrees_adt=node_degrees_adt,
                clustering_coeffs_rna=clustering_coeffs_rna,
                clustering_coeffs_adt=clustering_coeffs_adt
            )
        
        # Get training masks
        train_mask = rna_data.train_mask
        
        # Compute multi-task loss
        if aml_labels is not None:
            total_loss, adt_loss, aml_loss = compute_multi_task_loss(
                adt_pred[train_mask], adt_data.x[train_mask],
                aml_pred[train_mask], aml_labels[train_mask],
                adt_weight, classification_weight
            )
        else:
            # Fallback to ADT-only loss if no AML labels
            adt_loss = criterion(adt_pred[train_mask], adt_data.x[train_mask])
            aml_loss = torch.tensor(0.0, device=device)
            total_loss = adt_loss
        
        reg_loss = model.get_total_reg_loss()
        
        # Dynamic regularization scaling
        reg_lambda = 0.05 * (1 - epoch / epochs)
        total_loss = total_loss + reg_lambda * reg_loss
        
        # Backward pass
        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device.type == "cuda":
            print(f"⚠️ GPU OOM during training step {epoch}, clearing cache and retrying...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Fallback to CPU for this step
            device = torch.device('cpu')
            model = model.cpu()
            rna_data = rna_data.cpu()
            adt_data = adt_data.cpu()
            aml_labels = aml_labels.cpu() if aml_labels is not None else None
            adt_mean = adt_mean.cpu()
            adt_std = adt_std.cpu()
            node_degrees_rna = node_degrees_rna.cpu()
            node_degrees_adt = node_degrees_adt.cpu()
            clustering_coeffs_rna = clustering_coeffs_rna.cpu()
            clustering_coeffs_adt = clustering_coeffs_adt.cpu()
            print("🔄 Switched to CPU training due to GPU OOM")
            # Retry the training step on CPU
            return _training_step(
                model, rna_data, adt_data, aml_labels, optimizer, criterion, scaler,
                node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt,
                adt_mean, adt_std, adt_weight, classification_weight, epoch, epochs, False, device
            )
        elif "Found dtype Float but expected Half" in str(e) or "dtype" in str(e).lower():
            # Silently disable mixed precision and retry (no warning message)
            return _training_step(
                model, rna_data, adt_data, aml_labels, optimizer, criterion, scaler,
                node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt,
                adt_mean, adt_std, adt_weight, classification_weight, epoch, epochs, False, device
            )
        else:
            raise e
    
    return float(adt_loss.item()), float(reg_loss.item()), float(aml_loss.item())


def _evaluate_model(
    model, rna_data, adt_data, aml_labels, adt_mean, adt_std,
    node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt,
    mask, use_mixed_precision, device
) -> Dict[str, float]:
    """Evaluate model on a specific data split."""
    if mask.sum().item() == 0:
        return {k: float('nan') for k in ["MSE", "RMSE", "MAE", "R2", "MeanPearson", "MeanSpearman", 
                                         "AML_Accuracy", "AML_Precision", "AML_Recall", "AML_F1", "AML_AUC"]}
    
    model.eval()
    try:
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(use_mixed_precision and device.type == "cuda")):
            # Forward pass
            adt_pred, aml_pred, _ = model(
                x=rna_data.x,
                edge_index_rna=rna_data.edge_index,
                edge_index_adt=adt_data.edge_index if hasattr(adt_data, 'edge_index') else None,
                node_degrees_rna=node_degrees_rna,
                node_degrees_adt=node_degrees_adt,
                clustering_coeffs_rna=clustering_coeffs_rna,
                clustering_coeffs_adt=clustering_coeffs_adt
            )
    except RuntimeError as e:
        if "Found dtype Float but expected Half" in str(e) or "dtype" in str(e).lower():
            # Retry without mixed precision
            with torch.inference_mode():
                adt_pred, aml_pred, _ = model(
                    x=rna_data.x,
                    edge_index_rna=rna_data.edge_index,
                    edge_index_adt=adt_data.edge_index if hasattr(adt_data, 'edge_index') else None,
                    node_degrees_rna=node_degrees_rna,
                    node_degrees_adt=node_degrees_adt,
                    clustering_coeffs_rna=clustering_coeffs_rna,
                    clustering_coeffs_adt=clustering_coeffs_adt
                )
        else:
            raise e
    
    # Denormalize ADT predictions and targets
    adt_pred_denorm = adt_pred[mask] * adt_std + adt_mean
    adt_target = adt_data.x[mask] * adt_std + adt_mean
    
    # Convert to numpy for metrics
    adt_target_np = adt_target.detach().cpu().numpy()
    adt_pred_np = adt_pred_denorm.detach().cpu().numpy()
    
    # Get AML predictions and targets
    aml_pred_masked = aml_pred[mask]
    aml_target_masked = aml_labels[mask] if aml_labels is not None else None
    
    # Compute ADT regression metrics
    mse = mean_squared_error(adt_target_np, adt_pred_np)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(adt_target_np, adt_pred_np)
    r2 = r2_score(adt_target_np.reshape(-1), adt_pred_np.reshape(-1))
    
    # Compute per-marker correlations
    pearson_corrs, spearman_corrs = [], []
    for j in range(adt_target_np.shape[1]):
        yt = adt_target_np[:, j]
        yp = adt_pred_np[:, j]
        if np.std(yt) > 0 and np.std(yp) > 0:
            pearson_corrs.append(pearsonr(yt, yp)[0])
            spearman_corrs.append(spearmanr(yt, yp).correlation)
    
    mean_pearson = float(np.nanmean(pearson_corrs)) if pearson_corrs else float('nan')
    mean_spearman = float(np.nanmean(spearman_corrs)) if spearman_corrs else float('nan')
    
    # Compute AML classification metrics
    aml_metrics = {}
    if aml_target_masked is not None:
        aml_metrics = compute_classification_metrics(aml_pred_masked, aml_target_masked)
    else:
        aml_metrics = {
            'accuracy': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'f1_score': float('nan'),
            'auc_roc': float('nan')
        }
    
    return {
        "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2,
        "MeanPearson": mean_pearson, "MeanSpearman": mean_spearman,
        "AML_Accuracy": aml_metrics['accuracy'],
        "AML_Precision": aml_metrics['precision'],
        "AML_Recall": aml_metrics['recall'],
        "AML_F1": aml_metrics['f1_score'],
        "AML_AUC": aml_metrics['auc_roc']
    }


def _print_final_metrics(
    model, rna_data, adt_data, aml_labels, adt_mean, adt_std,
    node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt,
    use_mixed_precision, device
) -> None:
    """Print final metrics for all data splits."""
    print("\nFinal metrics:")
    
    splits = [
        ("Train", rna_data.train_mask),
        ("Val", rna_data.val_mask),
        ("Test", rna_data.test_mask)
    ]
    
    for split_name, mask in splits:
        metrics = _evaluate_model(
            model, rna_data, adt_data, aml_labels, adt_mean, adt_std,
            node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt,
            mask, use_mixed_precision, device
        )
        
        print(f"  {split_name:5s} | "
              f"MSE {metrics['MSE']:.6f}  RMSE {metrics['RMSE']:.6f}  MAE {metrics['MAE']:.6f}  "
              f"R² {metrics['R2']:.4f}  r_mean {metrics['MeanPearson']:.3f}  "
              f"ρ_mean {metrics['MeanSpearman']:.3f}")
        
        # Print AML classification metrics
        if not np.isnan(metrics['AML_Accuracy']):
            print(f"         | "
                  f"AML Acc {metrics['AML_Accuracy']:.3f}  Precision {metrics['AML_Precision']:.3f}  "
                  f"Recall {metrics['AML_Recall']:.3f}  F1 {metrics['AML_F1']:.3f}  "
                  f"AUC {metrics['AML_AUC']:.3f}")