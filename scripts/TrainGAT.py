def train_gat_model(data, model_name="GAT", epochs=200, use_cpu_fallback=False):
    """Train a GAT model and return the trained model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu_fallback else 'cpu')
    print(f"Using device: {device}")
    
    # Check memory requirements and adjust accordingly
    num_edges = data.num_edges
    num_nodes = data.num_nodes
    
    print(f"Graph stats - Nodes: {num_nodes}, Edges: {num_edges}")
    
    # Memory optimization: reduce model size if too many edges
    use_simple_model = False
    if num_edges > 2000000:  # If more than 2M edges
        print("Very large graph detected, using simplified GAT architecture...")
        hidden_dim = 32
        heads = 4
        use_simple_model = True
    elif num_edges > 1000000:  # If more than 1M edges
        print("Large graph detected, reducing model complexity...")
        hidden_dim = 32
        heads = 4
    else:
        hidden_dim = 64
        heads = 8
    
    # Create train/val/test masks
    N = data.num_nodes
    y_np = data.y.cpu().numpy()
    
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # Split 80/10/10
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, temp_idx = next(sss1.split(np.zeros(N), y_np))
    
    y_temp = y_np[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=43)
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
    
    # Initialize model
    in_dim = data.x.size(1)
    n_class = int(data.y.max().item() + 1)
    
    if use_simple_model:
        model = SimpleGAT(in_dim, hidden_dim, n_class, heads=heads).to(device)
        print(f"Using SimpleGAT: {in_dim} -> {n_class} (hidden: {hidden_dim}, heads: {heads})")
    else:
        model = GAT(in_dim, hidden_dim, n_class, heads=heads).to(device)
        print(f"Using GAT: {in_dim} -> {hidden_dim} -> {n_class} (heads: {heads})")
    
    # Move data to device with memory management
    cpu_fallback_triggered = False
    try:
        data = data.to(device)
        print(f"Successfully moved data to {device}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"GPU memory insufficient, falling back to CPU...")
            device = torch.device('cpu')
            model = model.cpu()
            data = data.cpu()
            cpu_fallback_triggered = True
        else:
            raise e
    
    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    def train():
        nonlocal model, data, optimizer, device, cpu_fallback_triggered
        
        model.train()
        optimizer.zero_grad()
        
        try:
            if device.type == 'cuda':
                torch.cuda.empty_cache()  # Clear cache before forward pass
            
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()  # Clear cache after backward pass
                
            return loss
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and not cpu_fallback_triggered:
                print(f"GPU OOM during training, switching to CPU...")
                # Move everything to CPU
                device = torch.device('cpu')
                model = model.cpu()
                data = data.cpu()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
                cpu_fallback_triggered = True
                
                # Retry the forward pass on CPU
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = criterion(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                return loss
            else:
                raise e
    
    def test(mask):
        model.eval()
        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct = pred[mask] == data.y[mask]
            acc = int(correct.sum()) / int(mask.sum())
            return acc
    
    print(f"Training {model_name} model...")
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        loss = train()
        
        if epoch % 50 == 0 or epoch == 1:
            val_acc = test(data.val_mask)
            test_acc = test(data.test_mask)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    final_test_acc = test(data.test_mask)
    print(f"Final {model_name} test accuracy: {final_test_acc:.4f}")
    
    return model, data


    # Train GAT models for both RNA and ADT with memory management
print("=== Training RNA GAT ===")
try:
    rna_gat_model, rna_data_with_masks = train_gat_model(rna_data, "RNA GAT", epochs=200)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("GPU memory insufficient for RNA GAT, trying CPU...")
        rna_gat_model, rna_data_with_masks = train_gat_model(rna_data, "RNA GAT", epochs=200, use_cpu_fallback=True)
    else:
        raise e

print("\n=== Training ADT GAT ===")
try:
    adt_gat_model, adt_data_with_masks = train_gat_model(adt_data, "ADT GAT", epochs=200)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("GPU memory insufficient for ADT GAT, trying CPU...")
        adt_gat_model, adt_data_with_masks = train_gat_model(adt_data, "ADT GAT", epochs=200, use_cpu_fallback=True)
    else:
        raise e

print("\n=== GAT Training Complete ===")
print(f"RNA GAT model trained successfully")
print(f"ADT GAT model trained successfully")


# Test the model
transformer_model.eval()
test_loss = 0.0
predictions = []
ground_truth = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = transformer_model(batch_x)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()
        
        predictions.append(outputs.cpu().numpy())
        ground_truth.append(batch_y.cpu().numpy())

test_loss /= len(test_loader)
predictions = np.vstack(predictions)
ground_truth = np.vstack(ground_truth)

# Calculate metrics
mse = mean_squared_error(ground_truth, predictions)
r2 = r2_score(ground_truth, predictions)

# Calculate correlation per dimension
pearson_corrs = []
spearman_corrs = []

for i in range(ground_truth.shape[1]):
    pearson_r, _ = pearsonr(ground_truth[:, i], predictions[:, i])
    spearman_r, _ = spearmanr(ground_truth[:, i], predictions[:, i])
    pearson_corrs.append(pearson_r)
    spearman_corrs.append(spearman_r)

mean_pearson = np.mean(pearson_corrs)
mean_spearman = np.mean(spearman_corrs)

print(f"\n=== Transformer Mapping Results ===")
print(f"Test Loss (MSE): {test_loss:.6f}")
print(f"MSE: {mse:.6f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Pearson Correlation: {mean_pearson:.4f}")
print(f"Mean Spearman Correlation: {mean_spearman:.4f}")