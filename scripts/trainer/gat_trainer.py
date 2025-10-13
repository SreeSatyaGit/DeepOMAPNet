import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.gat_models import SimpleGAT, GAT
from model.doNET import GATWithTransformerFusion

def train_gat_model(data, model_name="GAT", epochs=200, use_cpu_fallback=False, seed=42):
    """Train a GAT model and return the trained model"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if data.y.dtype != torch.long:
        data.y = data.y.long()

    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu_fallback else 'cpu')
    print(f"Using device: {device}")

    num_edges = getattr(data, "num_edges", None)
    num_nodes = getattr(data, "num_nodes", None)
    if num_edges is None or num_nodes is None:
        raise ValueError("`data` must have num_edges and num_nodes attributes.")
    print(f"Graph stats - Nodes: {num_nodes}, Edges: {num_edges}")

    use_simple_model = False
    if num_edges > 2_000_000:
        print("Very large graph detected, using simplified GAT architecture...")
        hidden_dim, heads = 32, 4
        use_simple_model = True
    elif num_edges > 1_000_000:
        print("Large graph detected, reducing model complexity...")
        hidden_dim, heads = 32, 4
    else:
        hidden_dim, heads = 64, 8

    N = data.num_nodes
    y_np = data.y.cpu().numpy()

    _, class_counts = np.unique(y_np, return_counts=True)
    if (class_counts < 2).any():
        raise ValueError(
            "Some classes have fewer than 2 samples; StratifiedShuffleSplit cannot proceed. "
            "Consider merging rare classes or switching to a random split."
        )

    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(N), y_np))

    y_temp = y_np[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
    val_rel, test_rel = next(sss2.split(np.zeros(len(temp_idx)), y_temp))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    in_dim = data.x.size(1)
    n_class = int(torch.unique(data.y).numel())

    if use_simple_model:
        model = SimpleGAT(in_dim, hidden_dim, n_class, heads=heads).to(device)
        print(f"Using SimpleGAT: {in_dim} -> {n_class} (hidden: {hidden_dim}, heads: {heads})")
    else:
        model = GAT(in_dim, hidden_dim, n_class, heads=heads).to(device)
        print(f"Using GAT: {in_dim} -> {hidden_dim} -> {n_class} (heads: {heads})")

    cpu_fallback_triggered = False
    try:
        data = data.to(device)
        print(f"Successfully moved data to {device}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU memory insufficient, falling back to CPU...")
            device = torch.device('cpu')
            model = model.cpu()
            data = data.cpu()
            cpu_fallback_triggered = True
        else:
            raise

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def _forward_backward():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss

    def train_step():
        nonlocal model, data, optimizer, device, cpu_fallback_triggered
        try:
            return _forward_backward()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and not cpu_fallback_triggered:
                print("GPU OOM during training, switching to CPU...")
                device = torch.device('cpu')
                model = model.cpu()
                data = data.cpu()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
                cpu_fallback_triggered = True
                return _forward_backward()
            raise

    def evaluate(mask):
        if mask.sum().item() == 0:
            return float("nan")
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred[mask] == data.y[mask]).sum().item()
            total = mask.sum().item()
            return correct / total

    print(f"Training {model_name} model...")
    best_val_acc = -1.0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        loss = train_step()
        if epoch % 50 == 0 or epoch == 1:
            val_acc = evaluate(data.val_mask)
            test_acc = evaluate(data.test_mask)
            print(f"Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
            if not np.isnan(val_acc) and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_test_acc = evaluate(data.test_mask)
    print(f"Final {model_name} test accuracy: {final_test_acc:.4f}")

    return model, data

def train_gat_transformer_fusion(rna_data, adt_data, epochs=200, use_cpu_fallback=False, seed=42):
    """
    Train GATWithTransformerFusion model for RNA to ADT mapping
    
    Args:
        rna_data: PyTorch Geometric data object for RNA
        adt_data: PyTorch Geometric data object for ADT (target)
        epochs: Number of training epochs
        use_cpu_fallback: Whether to use CPU if GPU fails
        seed: Random seed
    
    Returns:
        trained_model, rna_data, adt_data
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu_fallback else 'cpu')
    print(f"Using device: {device}")
    
    # Get data dimensions
    rna_input_dim = rna_data.x.size(1)
    adt_output_dim = adt_data.x.size(1)
    
    print(f"RNA input dimension: {rna_input_dim}")
    print(f"ADT output dimension: {adt_output_dim}")
    print(f"Number of nodes: {rna_data.num_nodes}")
    print(f"Number of edges: {rna_data.num_edges}")
    
    # Create train/val/test splits
    N = rna_data.num_nodes
    
    # Use stratified split based on some target (you might need to adjust this)
    # For now, use random split
    indices = torch.randperm(N)
    train_size = int(0.8 * N)
    val_size = int(0.1 * N)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    # Create masks
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    rna_data.train_mask = train_mask
    rna_data.val_mask = val_mask
    rna_data.test_mask = test_mask
    
    # Initialize model
    model = GATWithTransformerFusion(
        in_channels=rna_input_dim,
        hidden_channels=64,
        out_channels=adt_output_dim,
        heads=8,
        dropout=0.6,
        nhead=4,
        num_layers=2
    ).to(device)
    
    print(f"Model initialized: {rna_input_dim} -> {adt_output_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Move data to device
    cpu_fallback_triggered = False
    try:
        rna_data = rna_data.to(device)
        adt_data = adt_data.to(device)
        print(f"Successfully moved data to {device}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU memory insufficient, falling back to CPU...")
            device = torch.device('cpu')
            model = model.cpu()
            rna_data = rna_data.cpu()
            adt_data = adt_data.cpu()
            cpu_fallback_triggered = True
        else:
            raise
    
    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = torch.nn.MSELoss()
    
    def train_step():
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        adt_pred, fused_embeddings = model(
            x=rna_data.x,
            edge_index_rna=rna_data.edge_index,
            edge_index_adt=adt_data.edge_index if hasattr(adt_data, 'edge_index') else None
        )
        
        # Calculate loss on training set
        loss = criterion(adt_pred[rna_data.train_mask], adt_data.x[rna_data.train_mask])
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        return loss.item()
    
    def evaluate(mask):
        if mask.sum().item() == 0:
            return float("nan"), float("nan")
        
        model.eval()
        with torch.no_grad():
            adt_pred, fused_embeddings = model(
                x=rna_data.x,
                edge_index_rna=rna_data.edge_index,
                edge_index_adt=adt_data.edge_index if hasattr(adt_data, 'edge_index') else None
            )
            
            # Calculate MSE loss
            mse_loss = criterion(adt_pred[mask], adt_data.x[mask])
            
            # Calculate R² score
            y_true = adt_data.x[mask].cpu().numpy()
            y_pred = adt_pred[mask].cpu().numpy()
            
            # Flatten for R² calculation
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            
            # Calculate R²
            ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
            ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return mse_loss.item(), r2_score
    
    print("Training GATWithTransformerFusion model...")
    best_val_loss = float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    for epoch in range(1, epochs + 1):
        # Training step
        train_loss = train_step()
        train_losses.append(train_loss)
        
        # Evaluation
        if epoch % 10 == 0 or epoch == 1:
            val_loss, val_r2 = evaluate(rna_data.val_mask)
            test_loss, test_r2 = evaluate(rna_data.test_mask)
            
            val_losses.append(val_loss)
            val_r2_scores.append(val_r2)
            
            print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, Val R²: {val_r2:.4f}, "
                  f"Test Loss: {test_loss:.6f}, Test R²: {test_r2:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
    
    # Final evaluation
    final_test_loss, final_test_r2 = evaluate(rna_data.test_mask)
    print(f"Final test loss: {final_test_loss:.6f}")
    print(f"Final test R²: {final_test_r2:.4f}")
    
    return model, rna_data, adt_data
