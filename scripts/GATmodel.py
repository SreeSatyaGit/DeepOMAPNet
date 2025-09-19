class SimpleGAT(torch.nn.Module):
    """Simplified GAT for memory-constrained scenarios"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.6):
        super().__init__()
        self.dropout = dropout
        
        # Single GAT layer for memory efficiency
        self.conv1 = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=False)
        
    def forward(self, x, edge_index, return_embeddings=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        
        if return_embeddings:
            return x
        
        return x

    def get_embeddings(self, x, edge_index):
        """Get embeddings for mapping"""
        return self.forward(x, edge_index, return_embeddings=True)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super().__init__()
        self.dropout = dropout
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
        
    def forward(self, x, edge_index, return_embeddings=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_embeddings:
            return x  # Return embeddings before final layer
            
        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        """Get intermediate embeddings for mapping"""
        return self.forward(x, edge_index, return_embeddings=True)