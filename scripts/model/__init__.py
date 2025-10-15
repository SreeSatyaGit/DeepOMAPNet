from .transformer_models import TransformerMapping
from .doNET import (
    EnhancedGATWithTransformerFusion, 
    GATWithTransformerFusion,  # Backward compatibility alias
    EnhancedTransformerFusion,
    CrossAttentionLayer,
    SparseCrossAttentionLayer,  # New sparse attention layer
    GraphPositionalEncoding,
    AdapterLayer,
    compute_graph_statistics,
    compute_graph_statistics_fast
)

__all__ = [
    'TransformerMapping',
    'EnhancedGATWithTransformerFusion',  # Main enhanced model
    'GATWithTransformerFusion',          # Backward compatibility alias
    'EnhancedTransformerFusion',         # Enhanced fusion component
    'CrossAttentionLayer',               # Dense cross-attention component
    'SparseCrossAttentionLayer',         # Sparse cross-attention component
    'GraphPositionalEncoding',           # Positional encoding component
    'AdapterLayer',                      # Adapter component
    'compute_graph_statistics',          # Utility function (optimized)
    'compute_graph_statistics_fast',     # Ultra-fast utility function
]