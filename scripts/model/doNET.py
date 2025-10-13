import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch import nn

class TransformerFusion(nn.Module):
    def __init__(self, embedding_dim, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        # x shape: (num_nodes, embedding_dim)
        # transformer expects (sequence_length, batch_size, embedding_dim), treat nodes as sequence
        x = x.unsqueeze(1)  # (num_nodes, 1, embedding_dim)
        x = self.transformer_encoder(x)  # (num_nodes, 1, embedding_dim)
        return x.squeeze(1)  # (num_nodes, embedding_dim)

class GATWithTransformerFusion(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6, nhead=4, num_layers=2):
        super().__init__()
        self.dropout = dropout
        # GATRNA part - initial embedding extraction
        self.gat_rna1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat_rna2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        
        # Transformer fusion
        self.transformer_fusion = TransformerFusion(embedding_dim=hidden_channels, nhead=nhead, num_layers=num_layers, dropout=dropout)
        
        # GATADT part - output prediction from fused embeddings
        self.gat_adt = GATConv(hidden_channels, out_channels, heads=1, dropout=dropout)
        
    def forward(self, x, edge_index_rna, edge_index_adt=None):
        # Run GATRNA to get embeddings
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat_rna1(x, edge_index_rna)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        rna_embeddings = self.gat_rna2(x, edge_index_rna)
        rna_embeddings = F.elu(rna_embeddings)
        
        # Transformer fusion on RNA embeddings (and optional ADT embeddings if available)
        # Here, fusion is just on rna_embeddings, can extend to concat with ADT embeddings if given
        fused_embeddings = self.transformer_fusion(rna_embeddings)
        
        # Predict ADT expressions from fused embeddings, using GATADT layer
        # If ADT graph edges not given, reuse RNA edges or treat as fully connected in downstream experiments
        edge_index_adt = edge_index_adt if edge_index_adt is not None else edge_index_rna
        
        adt_pred = self.gat_adt(fused_embeddings, edge_index_adt)
        return adt_pred, fused_embeddings
    
    def get_embeddings(self, x, edge_index_rna):
        # Get learned embeddings after transformer fusion
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat_rna1(x, edge_index_rna)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        rna_embeddings = self.gat_rna2(x, edge_index_rna)
        rna_embeddings = F.elu(rna_embeddings)
        fused_embeddings = self.transformer_fusion(rna_embeddings)
        return fused_embeddings
