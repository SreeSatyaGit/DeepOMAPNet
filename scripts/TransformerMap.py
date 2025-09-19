class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerMapping(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=256, nhead=4, num_layers=3, dropout=0.1):
        super(TransformerMapping, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # Project input to transformer dimensions
        x = self.input_proj(x)
        
        # Add batch dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, d_model]
            
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Project to output dimensions
        x = self.output_proj(x.squeeze(1))
        
        return x

# Initialize transformer mapping model
import math  # For the positional encoding
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = rna_embeddings.shape[1]
output_dim = adt_embeddings.shape[1]

transformer_model = TransformerMapping(
    input_dim=input_dim, 
    output_dim=output_dim, 
    d_model=256,
    nhead=4,
    num_layers=3
).to(device)

print(f"Transformer Model: {input_dim} -> {output_dim}")
print(transformer_model)


# Convert embeddings to CPU and numpy
rna_emb_np = rna_embeddings.cpu().numpy()
adt_emb_np = adt_embeddings.cpu().numpy()

# Split data for training (use same train/val/test split as GAT)
train_mask_np = rna_data_with_masks.train_mask.cpu().numpy()
val_mask_np = rna_data_with_masks.val_mask.cpu().numpy()
test_mask_np = rna_data_with_masks.test_mask.cpu().numpy()

# Prepare training data
X_train = torch.tensor(rna_emb_np[train_mask_np], dtype=torch.float32)
y_train = torch.tensor(adt_emb_np[train_mask_np], dtype=torch.float32)

X_val = torch.tensor(rna_emb_np[val_mask_np], dtype=torch.float32)
y_val = torch.tensor(adt_emb_np[val_mask_np], dtype=torch.float32)

X_test = torch.tensor(rna_emb_np[test_mask_np], dtype=torch.float32)
y_test = torch.tensor(adt_emb_np[test_mask_np], dtype=torch.float32)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Create data loaders
batch_size = 64  # Smaller batch size for transformer
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Training parameters
num_epochs = 300
learning_rate = 0.0005  # Lower learning rate for transformer
weight_decay = 1e-4

# Warmup scheduler
from torch.optim.lr_scheduler import LambdaLR

def get_lr_scheduler(optimizer, warmup_steps=1000, max_steps=10000):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(max_steps - current_step) / float(max(1, max_steps - warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Training loop
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_state = None
patience = 30
patience_counter = 0

print("Training Transformer mapping model...")

for epoch in range(num_epochs):
    # Training phase
    transformer_model.train()
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = transformer_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation phase
    transformer_model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = transformer_model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = transformer_model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if (epoch + 1) % 25 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# Load best model/.,
transformer_model.load_state_dict(best_model_state)
print(f'Best validation loss: {best_val_loss:.6f}')