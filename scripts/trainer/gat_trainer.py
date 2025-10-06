import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.gat_models import SimpleGAT, GAT

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
