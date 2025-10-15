import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv
from torch_geometric.utils import softmax as segment_softmax
from torch_scatter import scatter_add
from torch import nn
import math

class GraphPositionalEncoding(nn.Module):
    """Graph-aware positional encoding based on node connectivity"""
    def __init__(self, embedding_dim, max_length=10000, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(max_length, embedding_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Graph structure-aware encoding
        self.degree_embedding = nn.Linear(1, embedding_dim // 4)
        self.clustering_embedding = nn.Linear(1, embedding_dim // 4)
        
    def forward(self, x, edge_index=None, node_degrees=None, clustering_coeffs=None):
        """
        Args:
            x: Node features [N, embedding_dim]
            edge_index: Graph edges [2, E] (optional)
            node_degrees: Node degrees [N] (optional)
            clustering_coeffs: Clustering coefficients [N] (optional)
        """
        N = x.size(0)
        
        # Basic positional encoding
        if N <= self.pos_embedding.size(0):
            pos_enc = self.pos_embedding[:N]
        else:
            # Interpolate for larger graphs
            pos_enc = F.interpolate(
                self.pos_embedding.unsqueeze(0).transpose(1, 2),
                size=N, mode='linear', align_corners=False
            ).squeeze(0).transpose(0, 1)
        
        # Add graph structure information with proper split:
        # base: d/2 | clustering: d/4 | degree: d/4
        base_dim = self.embedding_dim // 2
        quarter_dim = self.embedding_dim // 4
        base_enc = pos_enc[:, :base_dim]
        
        degree_enc = None
        clustering_enc = None
        if clustering_coeffs is not None:
            clustering_enc = self.clustering_embedding(clustering_coeffs.unsqueeze(-1).float())
        else:
            clustering_enc = torch.zeros(x.size(0), quarter_dim, device=x.device, dtype=x.dtype)
        if node_degrees is not None:
            degree_enc = self.degree_embedding(node_degrees.unsqueeze(-1).float())
        else:
            degree_enc = torch.zeros(x.size(0), quarter_dim, device=x.device, dtype=x.dtype)
        
        pos_enc = torch.cat([base_enc, clustering_enc, degree_enc], dim=-1)
        
        return self.dropout(x + pos_enc)

class SparseCrossAttentionLayer(nn.Module):
    """Truly sparse cross-attention layer using edge lists (no dense masks)"""
    def __init__(self, embedding_dim, nhead=8, dropout=0.1, use_positional_encoding=True, 
                 neighborhood_size=50):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.nhead = nhead
        self.head_dim = embedding_dim // nhead
        self.use_positional_encoding = use_positional_encoding
        self.neighborhood_size = neighborhood_size
        
        assert embedding_dim % nhead == 0, "embedding_dim must be divisible by nhead"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer normalization (pre-LN)
        self.norm_q = nn.LayerNorm(embedding_dim)
        self.norm_kv = nn.LayerNorm(embedding_dim)
        self.norm_out = nn.LayerNorm(embedding_dim)
        
        # Positional encoding for biological topology
        if use_positional_encoding:
            self.pos_encoding = GraphPositionalEncoding(embedding_dim, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def _preprocess_edges(self, edge_index, num_nodes, device):
        """Create symmetric, self-loop-augmented, optionally pruned edge list.
        Cached per-instance to avoid recomputation each forward.
        """
        if hasattr(self, '_cached_edges'):
            cached_num_nodes, cached_device, cached_edges = self._cached_edges
            if cached_num_nodes == num_nodes and cached_device == device:
                return cached_edges
        
        row, col = edge_index[0], edge_index[1]
        
        # Ensure symmetry (undirected): add (col,row)
        row_sym = torch.cat([row, col], dim=0)
        col_sym = torch.cat([col, row], dim=0)
        edge_index_sym = torch.stack([row_sym, col_sym], dim=0)
        
        # Add self-loops
        self_loops = torch.arange(num_nodes, device=device)
        loops = torch.stack([self_loops, self_loops], dim=0)
        edge_index_sym = torch.cat([edge_index_sym, loops], dim=1)
        
        # Do not deduplicate every forward; assume input is mostly clean.
        # If needed, dedup can be run offline.
        
        # Optional pruning to cap neighborhood size (deterministic: keep first K)
        if self.neighborhood_size < num_nodes:
            row, col = edge_index_sym[0], edge_index_sym[1]
            # Indices sorted by row to make slicing deterministic
            sort_idx = torch.argsort(row)
            row = row[sort_idx]
            col = col[sort_idx]
            
            # Compute starts via bincount cumsum
            deg = torch.bincount(row, minlength=num_nodes)
            starts = torch.zeros(num_nodes + 1, device=device, dtype=torch.long)
            starts[1:] = torch.cumsum(deg, dim=0)
            
            keep_mask = torch.ones(row.numel(), device=device, dtype=torch.bool)
            # Vectorized pruning: for nodes with deg>K, drop tail
            overfull = torch.where(deg > self.neighborhood_size)[0]
            if overfull.numel() > 0:
                idx_ranges = torch.stack([starts[overfull], starts[overfull] + deg[overfull]], dim=1)
                # mark tail beyond K as False
                for s, e in idx_ranges.tolist():
                    keep_mask[s + self.neighborhood_size:e] = False
            row = row[keep_mask]; col = col[keep_mask]
            edge_index_sym = torch.stack([row, col], dim=0)
        
        self._cached_edges = (num_nodes, device, edge_index_sym)
        return edge_index_sym
    
    def _sparse_attention_vectorized(self, q, k, v, edge_index):
        """
        Compute truly sparse attention using edge lists (vectorized, no dense masks)
        Args:
            q, k, v: [nhead, N, head_dim]
            edge_index: [2, E] edge list
        Returns:
            out: [nhead, N, head_dim]
            attn_weights: None (to save memory)
        """
        nhead, N, head_dim = q.shape
        device = q.device
        
        # Reshape for vectorized operations
        q = q.view(nhead, N, head_dim)  # [nhead, N, head_dim]
        k = k.view(nhead, N, head_dim)  # [nhead, N, head_dim]
        v = v.view(nhead, N, head_dim)  # [nhead, N, head_dim]
        
        # Flatten heads into edges for vectorized processing
        src = edge_index[0]  # [E]
        tgt = edge_index[1]  # [E]
        
        # Repeat edges for each head
        src_rep = src.unsqueeze(0).repeat(nhead, 1)  # [nhead, E]
        tgt_rep = tgt.unsqueeze(0).repeat(nhead, 1)  # [nhead, E]
        
        # Gather Q,K,V along edges for all heads
        q_src = q[torch.arange(nhead).unsqueeze(1), src_rep]  # [nhead, E, head_dim]
        k_tgt = k[torch.arange(nhead).unsqueeze(1), tgt_rep]  # [nhead, E, head_dim]
        v_tgt = v[torch.arange(nhead).unsqueeze(1), tgt_rep]  # [nhead, E, head_dim]
        
        # Compute scores and apply per-(head,src) softmax
        scores = (q_src * k_tgt).sum(dim=-1) * self.scale  # [nhead, E]
        # Build segment ids: head_offset + src
        head_offsets = (torch.arange(nhead, device=device) * (N + 1)).unsqueeze(1)  # [nhead,1]
        segment_ids = head_offsets + src_rep  # [nhead, E]
        attn = segment_softmax(scores.flatten(), segment_ids.flatten())  # [nhead*E]
        attn = attn.view(nhead, -1)  # [nhead, E]
        attn = self.dropout(attn)
        
        # Weighted sum via scatter_add over src for each head and feature dim
        out = torch.zeros(nhead, N, head_dim, device=device)
        # Expand attn for broadcasting
        attn_exp = attn.unsqueeze(-1)  # [nhead, E, 1]
        contrib = attn_exp * v_tgt  # [nhead, E, head_dim]
        
        # Scatter-add along src for each head
        for h in range(nhead):
            out[h] = scatter_add(contrib[h], src, dim=0, dim_size=N)
        
        return out, None
    
    def forward(self, query, key_value, edge_index=None, node_degrees=None, 
                clustering_coeffs=None, return_attention=False):
        """
        Args:
            query: RNA embeddings [N, embedding_dim]
            key_value: ADT embeddings [N, embedding_dim]
            edge_index: Graph edges [2, E] (optional)
            node_degrees: Node degrees [N] (optional)
            clustering_coeffs: Clustering coefficients [N] (optional)
            return_attention: Whether to return attention weights
        Returns:
            fused_embeddings: [N, embedding_dim]
            attention_weights: [nhead, N, N] (if return_attention=True)
        """
        # Apply positional encoding if enabled
        if self.use_positional_encoding:
            query = self.pos_encoding(query, edge_index, node_degrees, clustering_coeffs)
            key_value = self.pos_encoding(key_value, edge_index, node_degrees, clustering_coeffs)
        
        # Pre-normalization
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        
        # Project to Q, K, V
        q = self.q_proj(q)  # [N, embedding_dim]
        k = self.k_proj(kv)  # [N, embedding_dim]
        v = self.v_proj(kv)  # [N, embedding_dim]
        
        # Reshape for multi-head attention
        N = q.size(0)
        q = q.view(N, self.nhead, self.head_dim).transpose(0, 1)  # [nhead, N, head_dim]
        k = k.view(N, self.nhead, self.head_dim).transpose(0, 1)  # [nhead, N, head_dim]
        v = v.view(N, self.nhead, self.head_dim).transpose(0, 1)  # [nhead, N, head_dim]
        
        # Use vectorized sparse attention if edge_index is provided
        if edge_index is not None:
            edge_list = self._preprocess_edges(edge_index, N, q.device)
            # Use vectorized sparse attention
            out, attn_weights = self._sparse_attention_vectorized(q, k, v, edge_list)
        else:
            # Fallback to dense attention for small graphs
            if N < 1000:  # Only for small graphs
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [nhead, N, N]
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                out = torch.matmul(attn_weights, v)  # [nhead, N, head_dim]
            else:
                raise ValueError(f"Graph too large ({N} nodes) for dense attention. Provide edge_index for sparse attention.")
        
        # Reshape back
        out = out.transpose(0, 1).contiguous().view(N, self.embedding_dim)  # [N, embedding_dim]
        
        # Output projection and residual connection
        out = self.out_proj(out)
        out = self.norm_out(out + query)  # Residual connection with post-LN
        
        if return_attention:
            if attn_weights is None:
                # For sparse attention, return a placeholder indicating sparse attention was used
                return out, {"sparse_attention": True, "message": "Sparse attention used - no full attention matrix stored"}
            return out, attn_weights
        return out

# Keep the original CrossAttentionLayer for backward compatibility
class CrossAttentionLayer(nn.Module):
    """Original dense cross-attention layer (kept for backward compatibility)"""
    def __init__(self, embedding_dim, nhead=8, dropout=0.1, use_positional_encoding=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.nhead = nhead
        self.head_dim = embedding_dim // nhead
        self.use_positional_encoding = use_positional_encoding
        
        assert embedding_dim % nhead == 0, "embedding_dim must be divisible by nhead"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer normalization (pre-LN)
        self.norm_q = nn.LayerNorm(embedding_dim)
        self.norm_kv = nn.LayerNorm(embedding_dim)
        self.norm_out = nn.LayerNorm(embedding_dim)
        
        # Positional encoding for biological topology
        if use_positional_encoding:
            self.pos_encoding = GraphPositionalEncoding(embedding_dim, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, query, key_value, edge_index=None, node_degrees=None, 
                clustering_coeffs=None, return_attention=False):
        """
        Args:
            query: RNA embeddings [N, embedding_dim]
            key_value: ADT embeddings [N, embedding_dim]
            edge_index: Graph edges [2, E] (optional)
            node_degrees: Node degrees [N] (optional)
            clustering_coeffs: Clustering coefficients [N] (optional)
            return_attention: Whether to return attention weights
        Returns:
            fused_embeddings: [N, embedding_dim]
            attention_weights: [nhead, N, N] (if return_attention=True)
        """
        # Apply positional encoding if enabled
        if self.use_positional_encoding:
            query = self.pos_encoding(query, edge_index, node_degrees, clustering_coeffs)
            key_value = self.pos_encoding(key_value, edge_index, node_degrees, clustering_coeffs)
        
        # Pre-normalization
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        
        # Project to Q, K, V
        q = self.q_proj(q)  # [N, embedding_dim]
        k = self.k_proj(kv)  # [N, embedding_dim]
        v = self.v_proj(kv)  # [N, embedding_dim]
        
        # Reshape for multi-head attention
        N = q.size(0)
        q = q.view(N, self.nhead, self.head_dim).transpose(0, 1)  # [nhead, N, head_dim]
        k = k.view(N, self.nhead, self.head_dim).transpose(0, 1)  # [nhead, N, head_dim]
        v = v.view(N, self.nhead, self.head_dim).transpose(0, 1)  # [nhead, N, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [nhead, N, N]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights_dropped, v)  # [nhead, N, head_dim]
        
        # Reshape back
        out = out.transpose(0, 1).contiguous().view(N, self.embedding_dim)  # [N, embedding_dim]
        
        # Output projection and residual connection
        out = self.out_proj(out)
        out = self.norm_out(out + query)  # Residual connection with post-LN
        
        if return_attention:
            return out, attn_weights
        return out

class AdapterLayer(nn.Module):
    """Enhanced adapter layer with regularization and improved initialization"""
    def __init__(self, dim, reduction_factor=4, dropout=0.1, use_layernorm=True, 
                 adapter_l2_reg=1e-4, init_scale=0.1):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.adapter_l2_reg = adapter_l2_reg
        hidden_dim = max(dim // reduction_factor, 16)  # Ensure minimum hidden size
        
        # Pre-normalization (pre-LN scheme)
        if use_layernorm:
            self.norm = nn.LayerNorm(dim)
        
        # Down projection with careful initialization
        self.down = nn.Linear(dim, hidden_dim)
        nn.init.kaiming_normal_(self.down.weight, nonlinearity='relu')
        nn.init.zeros_(self.down.bias)
        
        # Up projection with near-zero initialization for stable residual learning
        self.up = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable scaling factor for residual connection
        self.scale = nn.Parameter(torch.ones(1) * init_scale)
        
        # Additional regularization layers
        self.activation_dropout = nn.Dropout(dropout * 0.5)
        
    def forward(self, x):
        identity = x
        
        # Pre-normalization if enabled
        if self.use_layernorm:
            x = self.norm(x)
        
        # Down projection with activation and dropout
        x = self.down(x)
        x = F.gelu(x)  # GELU for better gradient flow
        x = self.activation_dropout(x)
        
        # Up projection with scaled residual
        x = self.up(x)
        x = self.dropout(x)
        
        return identity + self.scale * x
    
    def get_l2_reg_loss(self):
        """Compute L2 regularization loss for adapter parameters"""
        l2_loss = 0.0
        for param in [self.down.weight, self.up.weight]:
            l2_loss += torch.norm(param, p=2) ** 2
        return self.adapter_l2_reg * l2_loss

class EnhancedTransformerFusion(nn.Module):
    def __init__(self, embedding_dim, nhead=8, num_layers=3, dropout=0.1, use_adapters=True,
                 reduction_factor=4, adapter_l2_reg=1e-4, use_positional_encoding=True,
                 use_sparse_attention=True, neighborhood_size=50):
        super().__init__()
        self.num_layers = num_layers
        self.use_adapters = use_adapters
        self.adapter_l2_reg = adapter_l2_reg
        self.use_positional_encoding = use_positional_encoding
        self.use_sparse_attention = use_sparse_attention
        
        # Cross-modal attention projection layers with better initialization
        self.rna_proj = nn.Linear(embedding_dim, embedding_dim)
        self.adt_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Initialize projection layers
        nn.init.xavier_uniform_(self.rna_proj.weight)
        nn.init.xavier_uniform_(self.adt_proj.weight)
        nn.init.zeros_(self.rna_proj.bias)
        nn.init.zeros_(self.adt_proj.bias)
        
        # Cross-attention layers for sophisticated fusion with positional encoding
        if use_sparse_attention:
            self.cross_attention_layers = nn.ModuleList([
                SparseCrossAttentionLayer(embedding_dim, nhead=nhead, dropout=dropout, 
                                         use_positional_encoding=use_positional_encoding,
                                         neighborhood_size=neighborhood_size)
                for _ in range(num_layers)
            ])
        else:
            self.cross_attention_layers = nn.ModuleList([
                CrossAttentionLayer(embedding_dim, nhead=nhead, dropout=dropout, 
                                   use_positional_encoding=use_positional_encoding)
                for _ in range(num_layers)
            ])
        
        # Graph transformer layers for each modality
        self.rna_transformer_layers = nn.ModuleList([
            TransformerConv(
                in_channels=embedding_dim,
                out_channels=embedding_dim // nhead,
                heads=nhead,
                dropout=dropout,
                edge_dim=None,
                concat=True
            ) for _ in range(num_layers)
        ])
        
        self.adt_transformer_layers = nn.ModuleList([
            TransformerConv(
                in_channels=embedding_dim,
                out_channels=embedding_dim // nhead,
                heads=nhead,
                dropout=dropout,
                edge_dim=None,
                concat=True
            ) for _ in range(num_layers)
        ])
        
        # Add adapters for parameter-efficient fine-tuning
        if use_adapters:
            self.adapters = nn.ModuleList([
                AdapterLayer(embedding_dim, reduction_factor=reduction_factor, 
                           dropout=dropout, adapter_l2_reg=adapter_l2_reg)
                for _ in range(num_layers)
            ])
        
        # Pre-LN normalization scheme
        self.rna_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_layers)
        ])
        self.adt_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, rna_x, adt_x, edge_index_rna, edge_index_adt=None, 
                node_degrees_rna=None, node_degrees_adt=None,
                clustering_coeffs_rna=None, clustering_coeffs_adt=None,
                return_attention=False):
        """
        Args:
            rna_x: RNA embeddings [N, embedding_dim]
            adt_x: ADT embeddings [N, embedding_dim]
            edge_index_rna: RNA graph edges [2, E_rna]
            edge_index_adt: ADT graph edges [2, E_adt] (optional)
            node_degrees_rna: RNA node degrees [N] (optional)
            node_degrees_adt: ADT node degrees [N] (optional)
            clustering_coeffs_rna: RNA clustering coefficients [N] (optional)
            clustering_coeffs_adt: ADT clustering coefficients [N] (optional)
            return_attention: Whether to return attention weights
        Returns:
            fused_embeddings: [N, embedding_dim]
            attention_weights: List of attention weights (if return_attention=True)
        """
        if edge_index_adt is None:
            edge_index_adt = edge_index_rna
            
        # Project both modalities
        rna_proj = self.rna_proj(rna_x)
        adt_proj = self.adt_proj(adt_x)
        
        # Store attention weights if requested
        attention_weights = [] if return_attention else None
        
        # Process through layers
        for i in range(self.num_layers):
            # Graph transformer for each modality
            rna_res = self.rna_transformer_layers[i](rna_proj, edge_index_rna)
            adt_res = self.adt_transformer_layers[i](adt_proj, edge_index_adt)
            
            # Apply adapters if enabled
            if self.use_adapters:
                rna_res = self.adapters[i](rna_res)
                adt_res = self.adapters[i](adt_res)
            
            # Pre-LN residual connections
            rna_proj = self.rna_norms[i](rna_proj + rna_res)
            adt_proj = self.adt_norms[i](adt_proj + adt_res)
            
            # Cross-attention fusion with positional encoding
            if return_attention:
                rna_fused, rna_attn = self.cross_attention_layers[i](
                    rna_proj, adt_proj, edge_index_rna, node_degrees_rna, 
                    clustering_coeffs_rna, return_attention=True
                )
                adt_fused, adt_attn = self.cross_attention_layers[i](
                    adt_proj, rna_proj, edge_index_adt, node_degrees_adt,
                    clustering_coeffs_adt, return_attention=True
                )
                attention_weights.append({
                    'rna_to_adt': rna_attn,
                    'adt_to_rna': adt_attn,
                    'layer': i
                })
            else:
                rna_fused = self.cross_attention_layers[i](
                    rna_proj, adt_proj, edge_index_rna, node_degrees_rna, 
                    clustering_coeffs_rna, return_attention=False
                )
                adt_fused = self.cross_attention_layers[i](
                    adt_proj, rna_proj, edge_index_adt, node_degrees_adt,
                    clustering_coeffs_adt, return_attention=False
                )
            
            # Update embeddings
            rna_proj = rna_proj + self.dropout(rna_fused)
            adt_proj = adt_proj + self.dropout(adt_fused)
        
        # Final fusion
        fused_embeddings = self.final_fusion(torch.cat([rna_proj, adt_proj], dim=-1))
        
        if return_attention:
            return fused_embeddings, attention_weights
        return fused_embeddings
    
    def get_adapter_reg_loss(self):
        """Compute total adapter regularization loss"""
        if not self.use_adapters:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        total_reg_loss = 0.0
        for adapter in self.adapters:
            total_reg_loss += adapter.get_l2_reg_loss()
        return total_reg_loss

class EnhancedGATWithTransformerFusion(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6, 
                 nhead=8, num_layers=3, use_adapters=True, reduction_factor=4, 
                 adapter_l2_reg=1e-4, use_positional_encoding=True, 
                 use_sparse_attention=True, neighborhood_size=50):
        super().__init__()
        self.dropout = dropout
        self.use_adapters = use_adapters
        self.adapter_l2_reg = adapter_l2_reg
        self.use_positional_encoding = use_positional_encoding
        self.use_sparse_attention = use_sparse_attention
        
        # RNA encoder with improved architecture
        self.gat_rna1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat_rna2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        
        # ADT encoder for multi-modal fusion
        self.gat_adt_init = GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout)
        
        # Enhanced transformer fusion with cross-attention and separate edge handling
        self.transformer_fusion = EnhancedTransformerFusion(
            embedding_dim=hidden_channels,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            use_adapters=use_adapters,
            reduction_factor=reduction_factor,
            adapter_l2_reg=adapter_l2_reg,
            use_positional_encoding=use_positional_encoding,
            use_sparse_attention=use_sparse_attention,
            neighborhood_size=neighborhood_size
        )
        
        # ADT prediction layers with better capacity control
        self.gat_adt = GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout)
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),  # Pre-LN
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        # Batch normalization for both modalities
        self.batch_norm_rna = nn.BatchNorm1d(hidden_channels)
        self.batch_norm_adt = nn.BatchNorm1d(hidden_channels)
        
        # Initialize final projection layers
        self._init_final_layers()
        
    def _init_final_layers(self):
        """Initialize final projection layers with proper scaling"""
        for module in self.final_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, x, edge_index_rna, edge_index_adt=None, return_attention=False,
                node_degrees_rna=None, node_degrees_adt=None,
                clustering_coeffs_rna=None, clustering_coeffs_adt=None):
        """
        Args:
            x: Input features [N, in_channels]
            edge_index_rna: RNA graph edges [2, E_rna]
            edge_index_adt: ADT graph edges [2, E_adt] (optional, uses RNA edges if None)
            return_attention: Whether to return attention weights
            node_degrees_rna: RNA node degrees [N] (optional)
            node_degrees_adt: ADT node degrees [N] (optional)
            clustering_coeffs_rna: RNA clustering coefficients [N] (optional)
            clustering_coeffs_adt: ADT clustering coefficients [N] (optional)
        Returns:
            adt_pred: ADT predictions [N, out_channels]
            fused_embeddings: Fused embeddings [N, hidden_channels]
            attention_weights: Attention weights (if return_attention=True)
        """
        # RNA embedding path
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat_rna1(x, edge_index_rna)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        rna_embeddings = self.gat_rna2(x, edge_index_rna)
        rna_embeddings = F.elu(rna_embeddings)
        rna_embeddings = self.batch_norm_rna(rna_embeddings)
        
        # Initial ADT embeddings from RNA
        edge_index_adt = edge_index_adt if edge_index_adt is not None else edge_index_rna
        initial_adt = self.gat_adt_init(rna_embeddings, edge_index_adt)
        initial_adt = F.elu(initial_adt)
        initial_adt = self.batch_norm_adt(initial_adt)
        
        # Multi-modal fusion with separate edge indices and positional encoding
        if return_attention:
            fused_embeddings, attention_weights = self.transformer_fusion(
                rna_embeddings, initial_adt, edge_index_rna, edge_index_adt,
                node_degrees_rna, node_degrees_adt,
                clustering_coeffs_rna, clustering_coeffs_adt,
                return_attention=True
            )
        else:
            fused_embeddings = self.transformer_fusion(
                rna_embeddings, initial_adt, edge_index_rna, edge_index_adt,
                node_degrees_rna, node_degrees_adt,
                clustering_coeffs_rna, clustering_coeffs_adt,
                return_attention=False
            )
        
        # ADT prediction with fused features
        adt_features = self.gat_adt(fused_embeddings, edge_index_adt)
        
        # Final prediction through MLP
        adt_pred = self.final_proj(adt_features)
        
        if return_attention:
            return adt_pred, fused_embeddings, attention_weights
        return adt_pred, fused_embeddings
    
    def get_embeddings(self, x, edge_index_rna, edge_index_adt=None,
                       node_degrees_rna=None, node_degrees_adt=None,
                       clustering_coeffs_rna=None, clustering_coeffs_adt=None):
        """Get embeddings for downstream tasks"""
        with torch.no_grad():
            x = F.dropout(x, p=self.dropout, training=False)
            x = self.gat_rna1(x, edge_index_rna)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=False)
            rna_embeddings = self.gat_rna2(x, edge_index_rna)
            rna_embeddings = F.elu(rna_embeddings)
            
            # Use separate edge indices if provided
            edge_index_adt = edge_index_adt if edge_index_adt is not None else edge_index_rna
            initial_adt = self.gat_adt_init(rna_embeddings, edge_index_adt)
            initial_adt = F.elu(initial_adt)
            
            fused_embeddings = self.transformer_fusion(
                rna_embeddings, initial_adt, edge_index_rna, edge_index_adt,
                node_degrees_rna, node_degrees_adt,
                clustering_coeffs_rna, clustering_coeffs_adt,
                return_attention=False
            )
            return fused_embeddings
    
    def get_total_reg_loss(self):
        """Get total regularization loss from adapters and projection layers"""
        reg_loss = self.transformer_fusion.get_adapter_reg_loss()
        
        # Add L2 regularization for projection layers
        for name, param in self.named_parameters():
            if 'proj' in name and 'weight' in name:
                reg_loss += self.adapter_l2_reg * torch.norm(param, p=2) ** 2
                
        return reg_loss

    
def compute_graph_statistics_fast(edge_index, num_nodes):
    """
    Simple and fast graph statistics computation
    
    Args:
        edge_index: Graph edges [2, E]
        num_nodes: Number of nodes
    
    Returns:
        node_degrees: Node degrees [N]
        clustering_coeffs: Simple degree-based clustering approximation [N]
    """
    device = edge_index.device
    
    # Compute node degrees efficiently
    node_degrees = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    unique_nodes, counts = torch.unique(edge_index[0], return_counts=True)
    node_degrees[unique_nodes] = counts.float()
    
    # Add incoming edges for symmetric graphs
    unique_nodes_in, counts_in = torch.unique(edge_index[1], return_counts=True)
    node_degrees[unique_nodes_in] += counts_in.float()
    
    # Simple clustering approximation: inverse of normalized degree
    max_degree = node_degrees.max()
    if max_degree > 0:
        normalized_degrees = node_degrees / max_degree
        clustering_coeffs = 0.5 * (1.0 - normalized_degrees) + 0.1  # Range [0.1, 0.6]
    else:
        clustering_coeffs = torch.full((num_nodes,), 0.3, device=device, dtype=torch.float32)
    
    return node_degrees, clustering_coeffs

# Backward compatibility alias
GATWithTransformerFusion = EnhancedGATWithTransformerFusion