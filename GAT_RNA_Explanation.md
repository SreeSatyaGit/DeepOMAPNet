# Graph Attention Networks (GAT) for RNA Data Analysis

## Introduction

Graph Attention Networks (GAT) are powerful deep learning models designed to work with graph-structured data. In the context of RNA sequencing data analysis, GATs can effectively model the relationships between cells based on their gene expression profiles. This document explains how GATs work and their application to RNA data.

## GAT Architecture for RNA Data

The GAT architecture used for RNA data analysis consists of multiple Graph Attention layers. Each layer applies attention mechanisms that enable nodes (cells) to aggregate information from their neighbors with different weights based on feature similarity.

![GAT Architecture](gat_architecture.png)
*Figure 1: GAT Architecture for RNA Data Analysis*

The GAT model processes RNA sequencing data as follows:

1. **Input**: RNA expression data (gene expression matrix) where each cell is represented as a node in a graph, and the edges between cells are determined by cell similarity (e.g., k-nearest neighbors).

2. **GATConv Layer 1**: The first layer applies multiple attention heads to learn which neighboring cells are most relevant for each target cell.

3. **ELU Activation**: A non-linear activation function (Exponential Linear Unit) is applied, followed by dropout for regularization.

4. **GATConv Layer 2**: The second layer further refines the cell representations.

5. **Output**: The final output can be used for cell type classification or as an embedding for further analysis.

```python
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_channels)
        self.conv2 = GATConv(hidden_channels, n_class)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

## Attention Mechanism

The core innovation in GAT is the attention mechanism that allows each node to focus on the most relevant neighbors. For RNA data, this means that each cell can learn to prioritize information from similar cell types while ignoring less relevant neighbors. The attention coefficient between two cells is computed based on their feature vectors (gene expression profiles).

![Attention Mechanism](attention_mechanism.png)
*Figure 2: Attention Mechanism in GAT - Different edge colors represent different attention weights*

Mathematically, the attention mechanism works as follows:

1. For each node pair (i,j) where j is a neighbor of i, compute attention coefficient e_ij = a(W·h_i, W·h_j) where a is an attention function, W is a learnable weight matrix, and h_i and h_j are feature vectors.

2. Normalize the coefficients using softmax: α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k exp(e_ik)

3. The updated feature for node i is then computed as: h'_i = σ(Σ_j α_ij · W·h_j) where σ is a non-linear activation.

## Multi-head Attention

To stabilize the learning process and enrich the model's representation capacity, GAT employs multi-head attention. This means running K independent attention mechanisms in parallel, and then concatenating or averaging their outputs. For RNA data, this allows the model to capture different aspects of cell-cell relationships.

![Multi-head Attention](multi_head_attention.png)
*Figure 3: Multi-head Attention in GAT - Multiple attention heads process the same input node*

With K attention heads, the output feature of node i in the l-th layer is:

h^l_i = ||^K_k=1 σ(Σ_j α^k_ij · W^k·h^{l-1}_j)

where || represents concatenation, α^k_ij is the normalized attention coefficient computed by the k-th attention head, and W^k is the corresponding weight matrix.

## Application to RNA Sequencing Data

When applied to RNA sequencing data, GAT offers several advantages:

1. **Cell Neighborhood Awareness**: Unlike standard neural networks, GAT considers the graph structure of cells, allowing it to leverage the similarity between cells in feature space.

2. **Adaptive Feature Aggregation**: The attention mechanism dynamically assigns different importance to different neighboring cells based on their feature similarity, which is crucial for identifying cell types.

3. **Interpretability**: The attention weights can be visualized to understand which connections between cells are most important, potentially revealing biological insights about cell-cell interactions.

4. **Robustness to Heterogeneity**: By focusing on relevant neighbors, GAT can handle the inherent heterogeneity in single-cell RNA sequencing data.

## Implementation Details

The GAT model for RNA data is implemented using PyTorch and PyTorch Geometric (PyG). The key components include:

1. **Data Preparation**: RNA expression data is preprocessed (normalization, log transformation, dimensionality reduction) and a cell similarity graph is constructed using k-nearest neighbors.

2. **Model Architecture**: The model consists of two GATConv layers with ELU activation and dropout for regularization.

3. **Training**: The model is trained using cross-entropy loss and Adam optimizer with weight decay for regularization.

4. **Evaluation**: The model's performance is evaluated based on classification accuracy on validation and test sets.

### Building a Cell Graph from RNA Data

```python
# Build a graph from RNA expression data
def build_graph_from_rna(adata):
    # Compute PCA if not already done
    if 'X_pca' not in adata.obsm:
        sc.pp.pca(adata)
    
    # Compute neighbors if not already done
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata)
    
    # Get adjacency matrix
    A = adata.obsp["connectivities"]
    
    # Convert to PyG edge_index format
    A_triu = sparse.triu(A, k=1)
    row, col = A_triu.nonzero()
    edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    
    # Get node features (PCA or integrated embeddings)
    X = (
        adata.obsm["X_integrated"] 
        if "X_integrated" in adata.obsm 
        else adata.obsm["X_pca"]
    )
    
    # Create PyG Data object
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
    )
    
    return data
```

## GAT for RNA-to-ADT Mapping

One powerful application of GAT is mapping between different molecular modalities, such as from RNA expression to Antibody-Derived Tags (ADT) protein expression. The process works as follows:

1. RNA expression data is used to construct a cell similarity graph.
2. GAT learns to aggregate information from similar cells.
3. The resulting cell embeddings capture the neighborhood structure of cells in gene expression space.
4. These embeddings can be used to predict ADT expression, enabling protein expression inference from RNA data alone.

### Implementation for RNA-to-ADT Mapping

```python
# GAT model for RNA-to-ADT mapping
class GATForRNAtoADT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x, edge_index):
        # First GAT layer with multi-head attention
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second GAT layer 
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Output projection to ADT space
        x = self.out_layer(x)
        
        return x
```

## Conclusion

Graph Attention Networks provide a powerful approach for analyzing RNA sequencing data by leveraging the graph structure of cells and their gene expression profiles. The attention mechanism allows the model to focus on the most relevant cell-cell relationships, leading to more accurate cell type identification and embedding. The multi-head attention design further enhances the model's ability to capture complex relationships in the data.

As single-cell genomics continues to advance, GAT and other graph neural network approaches will likely play an increasingly important role in extracting biological insights from high-dimensional sequencing data.
