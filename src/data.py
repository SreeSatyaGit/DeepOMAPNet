import torch
from torch.utils.data import Dataset

class CITESeqDataset(Dataset):
    """
    Dataset for CITE-seq data.
    - rna:       FloatTensor of shape (n_cells, n_genes)
    - proteins:  FloatTensor of shape (n_cells, n_proteins) (can be None)
    - prot_mask: BoolTensor of shape (n_cells, n_proteins) indicating observed proteins
    - labels:    LongTensor of shape (n_cells,) with cell type indices (can be None)
    - label_mask: BoolTensor of shape (n_cells,) indicating which cells have labels
    """
    def __init__(self, rna, proteins=None, prot_mask=None, labels=None, label_mask=None):
        self.rna = rna
        self.proteins = proteins
        self.prot_mask = prot_mask
        self.labels = labels
        self.label_mask = label_mask

    def __len__(self):
        return self.rna.size(0)

    def __getitem__(self, idx):
        x = self.rna[idx]
        prot = self.proteins[idx] if self.proteins is not None else torch.zeros_like(self.prot_mask[idx], dtype=torch.float32)
        p_mask = self.prot_mask[idx] if self.prot_mask is not None else torch.zeros_like(prot, dtype=torch.bool)
        lbl = self.labels[idx] if self.labels is not None else torch.tensor(0, dtype=torch.long)
        l_mask = self.label_mask[idx] if self.label_mask is not None else torch.tensor(False, dtype=torch.bool)
        return x, prot, p_mask, lbl, l_mask