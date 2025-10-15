import torch
import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

def train_gat_transformer_fusion(
    rna_data,
    adt_data,
    epochs=200,
    use_cpu_fallback=False,
    seed=42,
    stratify_labels=None,     # 1D array-like (n_nodes,) if you want stratified split
    train_frac=0.8,
    val_frac=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    dropout=0.4,  # Reduced from 0.6
    hidden_channels=96,  # Moderate increase from 64
    heads=8,
    nhead=8,  # Keep this increase
    num_layers=3,  # Moderate increase from 2
    amp=True,                 # mixed precision on CUDA
    patience=20,
):
    """
    Train GATWithTransformerFusion for RNA->ADT mapping with clean eval + history.

    Returns
    -------
    model, rna_data, adt_data, history
        history: dict with per-epoch (or every 10 epochs) metrics for plotting.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu_fallback else 'cpu')
    print(f"Using device: {device}")

    # Dimensions
    rna_input_dim = rna_data.x.size(1)
    adt_output_dim = adt_data.x.size(1)
    N = rna_data.num_nodes
    assert adt_data.num_nodes == N, "RNA and ADT must have same number of nodes (aligned cells)."

    # Create/validate masks (honor existing masks if present)
    if not hasattr(rna_data, "train_mask") or not hasattr(rna_data, "val_mask") or not hasattr(rna_data, "test_mask"):
        idx = np.arange(N)
        rng = np.random.default_rng(seed)
        if stratify_labels is not None:
            # stratified split: do it in two stages (train vs rest, then val vs test)
            from sklearn.model_selection import StratifiedShuffleSplit
            stratify_labels = np.asarray(stratify_labels)
            assert stratify_labels.shape[0] == N
            sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
            train_idx, rest_idx = next(sss1.split(idx, stratify_labels))
            rest_labels = stratify_labels[rest_idx]
            val_size = int(val_frac * N)
            test_size = N - len(train_idx) - val_size
            # split rest into val/test by stratify
            sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_size/len(rest_idx), random_state=seed)
            val_rel, test_rel = next(sss2.split(rest_idx, rest_labels))
            val_idx  = rest_idx[val_rel]
            test_idx = rest_idx[test_rel]
        else:
            rng.shuffle(idx)
            n_train = int(train_frac * N)
            n_val   = int(val_frac * N)
            train_idx = idx[:n_train]
            val_idx   = idx[n_train:n_train + n_val]
            test_idx  = idx[n_train + n_val:]

        train_mask = torch.zeros(N, dtype=torch.bool); train_mask[train_idx] = True
        val_mask   = torch.zeros(N, dtype=torch.bool); val_mask[val_idx]   = True
        test_mask  = torch.zeros(N, dtype=torch.bool); test_mask[test_idx] = True
        rna_data.train_mask = train_mask
        rna_data.val_mask   = val_mask
        rna_data.test_mask  = test_mask
    else:
        train_mask = rna_data.train_mask
        val_mask   = rna_data.val_mask
        test_mask  = rna_data.test_mask

    print(f"Splits — train: {int(train_mask.sum())}, val: {int(val_mask.sum())}, test: {int(test_mask.sum())}")

    # Init model
    from model.doNET import EnhancedGATWithTransformerFusion, compute_graph_statistics_fast
    model = EnhancedGATWithTransformerFusion(
        in_channels=rna_input_dim,
        hidden_channels=hidden_channels,
        out_channels=adt_output_dim,
        heads=heads,
        dropout=dropout,
        nhead=nhead,
        num_layers=num_layers,
        use_adapters=True,
        reduction_factor=4,
        adapter_l2_reg=5e-5,  # Reduced from 1e-4
        use_positional_encoding=True
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ADT preprocessing: CLR + z-score normalization (HUGE IMPACT for CITE-seq)
    print("Applying ADT preprocessing (CLR + z-score)...")
    
    # Step 1: CLR (centered log-ratio) per cell
    def clr_transform(x):
        x_pseudo = x + 1.0
        log_x = torch.log(x_pseudo)
        geometric_means = torch.exp(log_x.mean(dim=1, keepdim=True))
        x_clr = torch.log(x_pseudo / geometric_means)
        return x_clr
    
    # Apply CLR transformation
    adt_data.x = clr_transform(adt_data.x)
    print(f"CLR transformation applied: {adt_data.x.shape}")
    
    # Step 2: Z-score normalization per marker (column)
    adt_mean = adt_data.x.mean(dim=0, keepdim=True)
    adt_std = adt_data.x.std(dim=0, keepdim=True) + 1e-8  # Add small epsilon to avoid division by zero
    adt_data.x = (adt_data.x - adt_mean) / adt_std
    print(f"Target standardization applied - mean: {adt_mean.mean().item():.4f}, std: {adt_std.mean().item():.4f}")
    
    # Move data first, then compute graph statistics on the correct device
    try:
        rna_data = rna_data.to(device)
        adt_data = adt_data.to(device)
        print(f"Data moved to {device}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device.type == "cuda":
            print("GPU OOM → falling back to CPU.")
            device = torch.device('cpu'); model = model.cpu()
            rna_data = rna_data.cpu(); adt_data = adt_data.cpu()
        else:
            raise

    # Compute graph statistics for positional encoding (optimized) - after data is on correct device
    print("Computing graph statistics for positional encoding...")
    node_degrees_rna, clustering_coeffs_rna = compute_graph_statistics_fast(rna_data.edge_index, N)
    node_degrees_adt, clustering_coeffs_adt = compute_graph_statistics_fast(
        adt_data.edge_index if hasattr(adt_data, 'edge_index') else rna_data.edge_index, N
    )

    # Conservative optimizer settings - closer to original
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Back to MSE loss for now
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))
    
    # Reduced regularization scaling
    def get_reg_lambda(current_epoch):
        return 0.05 * (1 - current_epoch / epochs)  # start at 0.05, decrease to 0

    def _split_eval(mask):
        """Return dict: MSE, RMSE, MAE, R2, MeanPearson, MeanSpearman"""
        if mask.sum().item() == 0:
            return {k: float('nan') for k in ["MSE","RMSE","MAE","R2","MeanPearson","MeanSpearman"]}
        model.eval()
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(amp and device.type=="cuda")):
            y_pred, _ = model(
                x=rna_data.x,
                edge_index_rna=rna_data.edge_index,
                edge_index_adt=adt_data.edge_index if hasattr(adt_data, 'edge_index') else None,
                node_degrees_rna=node_degrees_rna,
                node_degrees_adt=node_degrees_adt,
                clustering_coeffs_rna=clustering_coeffs_rna,
                clustering_coeffs_adt=clustering_coeffs_adt
            )
            # Denormalize predictions and targets for proper metrics
            y_pred_denorm = y_pred[mask] * adt_std + adt_mean
            y_t = adt_data.x[mask] * adt_std + adt_mean  # Denormalize targets too
            y_t = y_t.detach().cpu().numpy()
            y_p = y_pred_denorm.detach().cpu().numpy()
        mse  = mean_squared_error(y_t, y_p)
        rmse = float(np.sqrt(mse))
        mae  = mean_absolute_error(y_t, y_p)
        # Flatten for global R2
        r2   = r2_score(y_t.reshape(-1), y_p.reshape(-1))
        # mean per-marker correlations
        pears, spears = [], []
        for j in range(y_t.shape[1]):
            yt = y_t[:, j]; yp = y_p[:, j]
            if np.std(yt) > 0 and np.std(yp) > 0:
                pears.append(pearsonr(yt, yp)[0])
                spears.append(spearmanr(yt, yp).correlation)
        mp = float(np.nanmean(pears)) if len(pears) else float('nan')
        ms = float(np.nanmean(spears)) if len(spears) else float('nan')
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MeanPearson": mp, "MeanSpearman": ms}

    def train_step():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(amp and device.type=="cuda")):
            y_pred, _ = model(
                x=rna_data.x,
                edge_index_rna=rna_data.edge_index,
                edge_index_adt=adt_data.edge_index if hasattr(adt_data, 'edge_index') else None,
                node_degrees_rna=node_degrees_rna,
                node_degrees_adt=node_degrees_adt,
                clustering_coeffs_rna=clustering_coeffs_rna,
                clustering_coeffs_adt=clustering_coeffs_adt
            )
            # Main prediction loss
            main_loss = criterion(y_pred[rna_data.train_mask], adt_data.x[rna_data.train_mask])
            
            # Add regularization loss with dynamic scaling
            reg_loss = model.get_total_reg_loss()
            lambda_reg = get_reg_lambda(epoch)
            total_loss = main_loss + lambda_reg * reg_loss
            
        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer); scaler.update()
        return float(main_loss.item()), float(reg_loss.item())

    print("Training...")
    best_val_r2 = float('-inf'); best_state = None; bad_epochs = 0
    history = {"epoch": [], "train_loss": [], "reg_loss": [], "val_MSE": [], "val_R2": [], "test_MSE": [], "test_R2": []}

    for epoch in range(1, epochs+1):
        train_loss, reg_loss = train_step()
        # monitor every 10 epochs (and at epoch 1)
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            val_metrics  = _split_eval(rna_data.val_mask)
            test_metrics = _split_eval(rna_data.test_mask)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["reg_loss"].append(reg_loss)
            history["val_MSE"].append(val_metrics["MSE"])
            history["val_R2"].append(val_metrics["R2"])
            history["test_MSE"].append(test_metrics["MSE"])
            history["test_R2"].append(test_metrics["R2"])

            print(f"Epoch {epoch:03d} | "
                  f"TrainLoss {train_loss:.6f} RegLoss {reg_loss:.6f} | "
                  f"Val MSE {val_metrics['MSE']:.6f} R² {val_metrics['R2']:.4f} | "
                  f"Test MSE {test_metrics['MSE']:.6f} R² {test_metrics['R2']:.4f}")

            # scheduler + early stopping on val R² (better metric)
            scheduler.step(val_metrics["MSE"])
            if val_metrics["R2"] > best_val_r2:
                best_val_r2 = val_metrics["R2"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch} (no val R² improvement for {patience} checks)")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model (val R²={best_val_r2:.4f}).")

    # Final report on all splits
    train_metrics = _split_eval(rna_data.train_mask)
    val_metrics   = _split_eval(rna_data.val_mask)
    test_metrics  = _split_eval(rna_data.test_mask)
    print("\nFinal metrics:")
    for name, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        print(f"  {name:5s} | MSE {m['MSE']:.6f}  RMSE {m['RMSE']:.6f}  MAE {m['MAE']:.6f}  "
              f"R² {m['R2']:.4f}  r_mean {m['MeanPearson']:.3f}  ρ_mean {m['MeanSpearman']:.3f}")

    return model, rna_data, adt_data, history
