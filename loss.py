import torch
import torch.nn.functional as F

def multitask_loss(prot_preds, prot_targets, prot_mask, class_logits, class_targets, class_mask, alpha=1.0):
    """
    Compute combined loss:
      - MSE for proteins (masked)
      - Cross-entropy for cell types (masked)
    alpha weighs classification loss relative to regression.
    """
    mse = ((prot_preds - prot_targets) ** 2 * prot_mask.float()).sum()
    mse = mse / prot_mask.sum().clamp_min(1.0)
    if class_mask.any():
        ce = F.cross_entropy(class_logits[class_mask], class_targets[class_mask])
    else:
        ce = torch.tensor(0.0, device=prot_preds.device)
    loss = mse + alpha * ce
    return loss, mse, ce