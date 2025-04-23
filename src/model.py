import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedEncoder(nn.Module):
    def __init__(self, input_dim, shared_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1000)
        self.fc2 = nn.Linear(1000, shared_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

class ProteinBranch(nn.Module):
    def __init__(self, shared_dim, branch_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(shared_dim, branch_dim)
        self.fc2 = nn.Linear(branch_dim, 1)

    def forward(self, h):
        h = F.relu(self.fc1(h))
        return self.fc2(h).squeeze(-1)

class CellTypeHead(nn.Module):
    def __init__(self, shared_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(shared_dim, num_classes)

    def forward(self, h):
        return self.fc(h)

class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, num_proteins, num_classes, shared_dim=128, branch_dim=64):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, shared_dim)
        self.protein_branches = nn.ModuleList([
            ProteinBranch(shared_dim, branch_dim)
            for _ in range(num_proteins)
        ])
        self.classifier = CellTypeHead(shared_dim, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        prot_preds = torch.stack([b(h) for b in self.protein_branches], dim=1)
        class_logits = self.classifier(h)
        return prot_preds, class_logits