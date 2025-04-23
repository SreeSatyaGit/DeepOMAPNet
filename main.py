import torch
from data import CITESeqDataset
from model import MultiTaskModel
from train import train

if __name__ == "__main__":
    # Hyperparameters
    INPUT_DIM   = 2000
    N_PROTEINS  = 20
    N_CLASSES   = 10
    SHARED_DIM  = 128
    BRANCH_DIM  = 64
    BATCH_SIZE  = 128
    LR          = 1e-3
    EPOCHS      = 30
    ALPHA       = 1.0
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Replace with real data
    X_rna     = torch.randn(5000, INPUT_DIM)
    Y_prot    = torch.randn(5000, N_PROTEINS)
    mask_prot = torch.rand(5000, N_PROTEINS) > 0.2
    Y_labels  = torch.randint(0, N_CLASSES, (5000,))
    mask_lbl  = torch.rand(5000) > 0.5

    dataset = CITESeqDataset(X_rna, Y_prot, mask_prot, Y_labels, mask_lbl)
    model   = MultiTaskModel(INPUT_DIM, N_PROTEINS, N_CLASSES, SHARED_DIM, BRANCH_DIM)

    train(model, dataset, BATCH_SIZE, EPOCHS, LR, ALPHA, DEVICE)