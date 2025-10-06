import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerMapping(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=256, nhead=4, num_layers=3, dropout=0.1):
        super(TransformerMapping, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        
        # Ensure correct shape for transformer (batch_first=True expects [batch, seq, features])
        if len(x.shape) == 2:
            # Add sequence dimension: [batch, features] -> [batch, 1, features]
            x = x.unsqueeze(1)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Remove sequence dimension and project to output
        if x.shape[1] == 1:
            x = x.squeeze(1)
        x = self.output_proj(x)
        
        return x