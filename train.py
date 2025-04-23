import torch
from torch.utils.data import DataLoader
from loss import multitask_loss


def train(model, dataset, batch_size, epochs, lr, alpha, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_mse, total_ce = 0.0, 0.0, 0.0
        for x, prot, p_mask, lbl, l_mask in loader:
            x, prot = x.to(device), prot.to(device)
            p_mask, lbl = p_mask.to(device), lbl.to(device)
            l_mask = l_mask.to(device)
            optimizer.zero_grad()
            prot_preds, class_logits = model(x)
            loss, mse, ce = multitask_loss(prot_preds, prot, p_mask, class_logits, lbl, l_mask, alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_mse  += mse.item()
            total_ce   += ce.item()
        n = len(loader)
        print(f"Epoch {epoch:02d} | Loss: {total_loss/n:.4f} | MSE: {total_mse/n:.4f} | CE: {total_ce/n:.4f}")
