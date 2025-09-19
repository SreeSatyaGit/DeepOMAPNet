def sparsify_graph(adata, max_edges_per_node=50):
    """Sparsify the graph by keeping only top k neighbors per node"""
    
    # Check if connectivities exists
    if "connectivities" not in adata.obsp:
        print("No connectivity graph found. Computing neighbors first...")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    
    A = adata.obsp["connectivities"].tocsr()
    n_nodes = A.shape[0]
    
    # Check if sparsification is needed
    avg_degree = A.nnz / n_nodes
    if avg_degree <= max_edges_per_node:
        print(f"Graph already sparse enough (avg degree: {avg_degree:.1f})")
        return adata
    
    print(f"Sparsifying graph from avg degree {avg_degree:.1f} to max {max_edges_per_node}")
    
    # Create new sparse matrix
    row_indices = []
    col_indices = []
    data_values = []
    
    for i in range(n_nodes):
        # Get neighbors and their weights for node i
        start_idx = A.indptr[i]
        end_idx = A.indptr[i + 1]
        neighbors = A.indices[start_idx:end_idx]
        weights = A.data[start_idx:end_idx]
        
        # Keep only top k neighbors
        if len(neighbors) > max_edges_per_node:
            top_k_indices = np.argpartition(weights, -max_edges_per_node)[-max_edges_per_node:]
            neighbors = neighbors[top_k_indices]
            weights = weights[top_k_indices]
        
        # Add edges
        row_indices.extend([i] * len(neighbors))
        col_indices.extend(neighbors)
        data_values.extend(weights)
    
    # Create new adjacency matrix
    A_sparse = sparse.csr_matrix(
        (data_values, (row_indices, col_indices)), 
        shape=(n_nodes, n_nodes)
    )
    
    # Make symmetric
    A_sparse = (A_sparse + A_sparse.T) / 2
    
    # Update the AnnData object
    adata.obsp["connectivities"] = A_sparse
    
    new_avg_degree = A_sparse.nnz / n_nodes
    print(f"New average degree: {new_avg_degree:.1f}")
    
    return adata

def build_pyg_data(adata, use_pca=True, sparsify_large_graphs=True, max_edges_per_node=50):
    """Build PyTorch Geometric Data object from AnnData"""
    
    # Ensure PCA and neighbor graph are computed
    if use_pca and "X_pca" not in adata.obsm:
        print("Computing PCA first...")
        sc.tl.pca(adata, n_comps=50, svd_solver="arpack")
    
    if "connectivities" not in adata.obsp:
        print("Computing neighbor graph first...")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50 if use_pca else None)
        
    if "leiden" not in adata.obs:
        print("Computing leiden clusters first...")
        sc.tl.leiden(adata, resolution=1.0)
    
    # Sparsify if needed
    if sparsify_large_graphs:
        A = adata.obsp["connectivities"]
        avg_degree = A.nnz / A.shape[0]
        if avg_degree > max_edges_per_node:
            print(f"Large graph detected (avg degree: {avg_degree:.1f}), applying sparsification...")
            adata = sparsify_graph(adata, max_edges_per_node)
    
    # Features
    X = adata.obsm["X_pca"] if use_pca else adata.X.toarray()
    
    # Labels (leiden clusters)
    y = adata.obs["leiden"].astype(int).to_numpy()
    
    # Edge index from connectivities
    A = adata.obsp["connectivities"].tocsr()
    A_triu = sparse.triu(A, k=1)
    row, col = A_triu.nonzero()
    edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    
    # Create PyG Data object
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
    )
    
    return data

# Build data objects with memory optimization
print("Building PyG data objects...")

# Check available GPU memory
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    print(f"Available GPU memory: {gpu_memory:.1f} GB")
    
    # Ensure neighbors are computed for both datasets first
    if "connectivities" not in trainGene.obsp:
        print("Computing neighbors for RNA data...")
        sc.pp.neighbors(trainGene, n_neighbors=15, n_pcs=50)
        
    if "connectivities" not in trainADT.obsp:
        print("Computing neighbors for ADT data...")
        sc.pp.neighbors(trainADT, n_neighbors=15, n_pcs=50)
    
    # Now it's safe to get the number of edges
    rna_edges = trainGene.obsp["connectivities"].nnz
    adt_edges = trainADT.obsp["connectivities"].nnz
    
    print(f"RNA graph edges: {rna_edges:,}")
    print(f"ADT graph edges: {adt_edges:,}")
    
    # Set sparsification based on graph size
    max_edges_rna = 100 if rna_edges > 5000000 else 200
    max_edges_adt = 50 if adt_edges > 10000000 else 100
    
    print(f"Using max edges per node - RNA: {max_edges_rna}, ADT: {max_edges_adt}")
else:
    print("Using CPU - no memory constraints")
    max_edges_rna = 200
    max_edges_adt = 100

# Build data objects
rna_data = build_pyg_data(trainGene, use_pca=True, sparsify_large_graphs=True, max_edges_per_node=max_edges_rna)
adt_data = build_pyg_data(trainADT, use_pca=True, sparsify_large_graphs=True, max_edges_per_node=max_edges_adt)

print(f"RNA PyG data - Nodes: {rna_data.num_nodes}, Edges: {rna_data.num_edges}, Features: {rna_data.num_node_features}")
print(f"ADT PyG data - Nodes: {adt_data.num_nodes}, Edges: {adt_data.num_edges}, Features: {adt_data.num_node_features}")



def extract_embeddings(model, data):
    """Extract embeddings from trained GAT model"""
    model.eval()
    
    # Ensure model and data are on the same device
    device = next(model.parameters()).device
    if data.x.device != device:
        print(f"Moving data from {data.x.device} to {device}")
        data = data.to(device)
    
    with torch.no_grad():
        # Clear cache if using GPU
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        embeddings = model.get_embeddings(data.x, data.edge_index)
        
        # Move to CPU for further processing
        embeddings = embeddings.cpu()
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    return embeddings

# Clear any existing cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Extract embeddings
print("Extracting embeddings...")
rna_embeddings = extract_embeddings(rna_gat_model, rna_data_with_masks)
adt_embeddings = extract_embeddings(adt_gat_model, adt_data_with_masks)

print(f"RNA embeddings shape: {rna_embeddings.shape}")
print(f"ADT embeddings shape: {adt_embeddings.shape}")

# Ensure both embeddings have the same number of cells
assert rna_embeddings.shape[0] == adt_embeddings.shape[0], "Number of cells must match"