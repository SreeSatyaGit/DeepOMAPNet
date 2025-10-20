"""
Standardized Visualization Module for Single-Cell Multi-Omics GAT Analysis

This module provides centralized, reproducible visualization functions for
single-cell RNA-seq and CITE-seq data analysis using Graph Attention Networks (GAT).

Author: Computational Biology Lab
Date: 2025-10-20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import umap
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import pearsonr, spearmanr
import torch
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Color palettes
DISEASE_COLORS = {
    'AML': '#E74C3C', 'Healthy': '#3498DB', 'Normal': '#3498DB',
    '0': '#3498DB', '1': '#E74C3C',  # For numeric labels
    0: '#3498DB', 1: '#E74C3C'  # For integer labels
}
CLUSTER_COLORS = sns.color_palette('tab20', 20)


def plot_umap_rna_embeddings(rna_embeddings, cell_labels=None, cell_types=None,
                             title='UMAP of RNA GAT Embeddings',
                             save_path='umap_rna_embeddings.pdf'):
    """
    Plot UMAP visualization of RNA GAT embeddings.
    
    Args:
        rna_embeddings: [N_cells, embedding_dim] - RNA embeddings from GAT
        cell_labels: [N_cells] - Disease labels (AML vs Healthy)
        cell_types: [N_cells] - Cell type annotations (optional)
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        fig: Matplotlib figure object
        umap_coords: UMAP coordinates for further analysis
    """
    print(f"📊 Generating UMAP for RNA embeddings ({rna_embeddings.shape[0]} cells)...")
    
    # Convert to numpy if tensor
    if torch.is_tensor(rna_embeddings):
        rna_embeddings = rna_embeddings.cpu().numpy()
    
    # Compute UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    umap_coords = reducer.fit_transform(rna_embeddings)
    
    # Create figure
    fig, axes = plt.subplots(1, 2 if cell_types is not None else 1, 
                            figsize=(16 if cell_types is not None else 10, 8))
    
    if cell_types is None:
        axes = [axes]
    
    # Plot 1: Color by disease status
    ax1 = axes[0]
    if cell_labels is not None:
        # Convert to numpy if needed
        if torch.is_tensor(cell_labels):
            cell_labels = cell_labels.cpu().numpy()
        
        unique_labels = np.unique(cell_labels)
        print(f"📊 Found {len(unique_labels)} unique labels: {unique_labels}")
        
        # Define colors for each label
        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
        label_names = ['Normal/Healthy', 'AML', 'Other1', 'Other2', 'Other3', 'Other4']
        
        for i, label in enumerate(unique_labels):
            mask = cell_labels == label
            color = colors[i % len(colors)]  # Cycle through colors if more labels than colors
            label_name = label_names[i % len(label_names)] if i < len(label_names) else f'Class_{label}'
            
            print(f"  Label {label} ({label_name}): {mask.sum()} cells, color: {color}")
            
            ax1.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=color, label=f'{label_name} ({mask.sum()})', s=3, alpha=0.6)
        ax1.legend(markerscale=3, frameon=True, loc='best')
    else:
        ax1.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                   c='#3498DB', s=3, alpha=0.6)
    
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_title('RNA Embeddings - Disease Status')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Color by cell type (if available)
    if cell_types is not None:
        ax2 = axes[1]
        unique_types = np.unique(cell_types)
        colors = sns.color_palette('tab20', len(unique_types))
        
        for i, cell_type in enumerate(unique_types):
            mask = cell_types == cell_type
            ax2.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[colors[i]], label=str(cell_type), s=3, alpha=0.6)
        
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.set_title('RNA Embeddings - Cell Types')
        ax2.legend(markerscale=3, frameon=True, loc='best', ncol=2)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    
    plt.show()
    
    return fig, umap_coords


def plot_umap_adt_embeddings(adt_embeddings, cell_labels=None, cell_types=None,
                             title='UMAP of ADT GAT Embeddings',
                             save_path='umap_adt_embeddings.pdf'):
    """
    Plot UMAP visualization of ADT GAT embeddings.
    
    Args:
        adt_embeddings: [N_cells, embedding_dim] - ADT embeddings from GAT
        cell_labels: [N_cells] - Disease labels (AML vs Healthy)
        cell_types: [N_cells] - Cell type annotations (optional)
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        fig: Matplotlib figure object
        umap_coords: UMAP coordinates for further analysis
    """
    print(f"📊 Generating UMAP for ADT embeddings ({adt_embeddings.shape[0]} cells)...")
    
    # Convert to numpy if tensor
    if torch.is_tensor(adt_embeddings):
        adt_embeddings = adt_embeddings.cpu().numpy()
    
    # Compute UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    umap_coords = reducer.fit_transform(adt_embeddings)
    
    # Create figure
    fig, axes = plt.subplots(1, 2 if cell_types is not None else 1,
                            figsize=(16 if cell_types is not None else 10, 8))
    
    if cell_types is None:
        axes = [axes]
    
    # Plot 1: Color by disease status
    ax1 = axes[0]
    if cell_labels is not None:
        # Convert to numpy if needed
        if torch.is_tensor(cell_labels):
            cell_labels = cell_labels.cpu().numpy()
        
        unique_labels = np.unique(cell_labels)
        print(f"📊 Found {len(unique_labels)} unique labels: {unique_labels}")
        
        # Define colors for each label
        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
        label_names = ['Normal/Healthy', 'AML', 'Other1', 'Other2', 'Other3', 'Other4']
        
        for i, label in enumerate(unique_labels):
            mask = cell_labels == label
            color = colors[i % len(colors)]  # Cycle through colors if more labels than colors
            label_name = label_names[i % len(label_names)] if i < len(label_names) else f'Class_{label}'
            
            print(f"  Label {label} ({label_name}): {mask.sum()} cells, color: {color}")
            
            ax1.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=color, label=f'{label_name} ({mask.sum()})', s=3, alpha=0.6)
        ax1.legend(markerscale=3, frameon=True, loc='best')
    else:
        ax1.scatter(umap_coords[:, 0], umap_coords[:, 1],
                   c='#E74C3C', s=3, alpha=0.6)
    
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_title('ADT Embeddings - Disease Status')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Color by cell type (if available)
    if cell_types is not None:
        ax2 = axes[1]
        unique_types = np.unique(cell_types)
        colors = sns.color_palette('tab20', len(unique_types))
        
        for i, cell_type in enumerate(unique_types):
            mask = cell_types == cell_type
            ax2.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[colors[i]], label=str(cell_type), s=3, alpha=0.6)
        
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.set_title('ADT Embeddings - Cell Types')
        ax2.legend(markerscale=3, frameon=True, loc='best', ncol=2)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    
    plt.show()
    
    return fig, umap_coords


def plot_umap_fused_embeddings(fused_embeddings, cell_labels=None, cell_types=None,
                               title='UMAP of Fused Multimodal Embeddings',
                               save_path='umap_fused_embeddings.pdf'):
    """
    Plot UMAP visualization of fused (RNA+ADT) GAT embeddings.
    
    Args:
        fused_embeddings: [N_cells, embedding_dim] - Fused embeddings from transformer fusion
        cell_labels: [N_cells] - Disease labels (AML vs Healthy)
        cell_types: [N_cells] - Cell type annotations (optional)
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        fig: Matplotlib figure object
        umap_coords: UMAP coordinates for further analysis
    """
    print(f"📊 Generating UMAP for fused embeddings ({fused_embeddings.shape[0]} cells)...")
    
    # Convert to numpy if tensor
    if torch.is_tensor(fused_embeddings):
        fused_embeddings = fused_embeddings.cpu().numpy()
    
    # Compute UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    umap_coords = reducer.fit_transform(fused_embeddings)
    
    # Create figure
    fig, axes = plt.subplots(1, 2 if cell_types is not None else 1,
                            figsize=(16 if cell_types is not None else 10, 8))
    
    if cell_types is None:
        axes = [axes]
    
    # Plot 1: Color by disease status
    ax1 = axes[0]
    if cell_labels is not None:
        # Convert to numpy if needed
        if torch.is_tensor(cell_labels):
            cell_labels = cell_labels.cpu().numpy()
        
        unique_labels = np.unique(cell_labels)
        print(f"📊 Found {len(unique_labels)} unique labels: {unique_labels}")
        
        # Define colors for each label
        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
        label_names = ['Normal/Healthy', 'AML', 'Other1', 'Other2', 'Other3', 'Other4']
        
        for i, label in enumerate(unique_labels):
            mask = cell_labels == label
            color = colors[i % len(colors)]  # Cycle through colors if more labels than colors
            label_name = label_names[i % len(label_names)] if i < len(label_names) else f'Class_{label}'
            
            print(f"  Label {label} ({label_name}): {mask.sum()} cells, color: {color}")
            
            ax1.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=color, label=f'{label_name} ({mask.sum()})', s=3, alpha=0.6)
        ax1.legend(markerscale=3, frameon=True, loc='best')
    else:
        ax1.scatter(umap_coords[:, 0], umap_coords[:, 1],
                   c='#9B59B6', s=3, alpha=0.6)
    
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_title('Fused Embeddings - Disease Status')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Color by cell type (if available)
    if cell_types is not None:
        ax2 = axes[1]
        unique_types = np.unique(cell_types)
        colors = sns.color_palette('tab20', len(unique_types))
        
        for i, cell_type in enumerate(unique_types):
            mask = cell_types == cell_type
            ax2.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[colors[i]], label=str(cell_type), s=3, alpha=0.6)
        
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.set_title('Fused Embeddings - Cell Types')
        ax2.legend(markerscale=3, frameon=True, loc='best', ncol=2)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    
    plt.show()
    
    return fig, umap_coords


def plot_adt_marker_correlations(adt_true, adt_pred, marker_names=None,
                                 markers_to_plot=None,
                                 save_path='adt_marker_correlations.pdf'):
    """
    Plot correlation between true and predicted ADT values for specific markers.
    
    Args:
        adt_true: [N_cells, N_markers] - True ADT values
        adt_pred: [N_cells, N_markers] - Predicted ADT values
        marker_names: List of marker names
        markers_to_plot: List of specific markers to plot (default: CD19, CD3, CD34, cMPO, CD45, CD79a, CD7)
        save_path: Path to save figure
    
    Returns:
        fig: Matplotlib figure object
        correlations: Dictionary with correlation statistics per marker
    """
    print(f"📊 Generating ADT marker correlation plots...")
    
    # Convert to numpy if tensor
    if torch.is_tensor(adt_true):
        adt_true = adt_true.cpu().numpy()
    if torch.is_tensor(adt_pred):
        adt_pred = adt_pred.cpu().numpy()
    
    # Default markers to plot
    if markers_to_plot is None:
        markers_to_plot = ['CD19', 'CD3', 'CD34', 'cMPO', 'CD45', 'CD79a', 'CD7']
    
    # Find marker indices
    marker_indices = []
    marker_names_found = []
    
    if marker_names is not None:
        for marker in markers_to_plot:
            # Find marker in list (case insensitive, partial match)
            for i, name in enumerate(marker_names):
                if marker.lower() in name.lower():
                    marker_indices.append(i)
                    marker_names_found.append(name)
                    break
    else:
        # Use first 7 markers if no names provided
        marker_indices = list(range(min(7, adt_true.shape[1])))
        marker_names_found = [f'Marker_{i}' for i in marker_indices]
    
    n_markers = len(marker_indices)
    if n_markers == 0:
        print("⚠️ No matching markers found!")
        return None, {}
    
    # Create figure
    ncols = 4
    nrows = int(np.ceil(n_markers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = axes.flatten() if n_markers > 1 else [axes]
    
    correlations = {}
    
    for idx, (marker_idx, marker_name) in enumerate(zip(marker_indices, marker_names_found)):
        ax = axes[idx]
        
        # Get data for this marker
        true_vals = adt_true[:, marker_idx]
        pred_vals = adt_pred[:, marker_idx]
        
        # Calculate correlations
        pearson_r, pearson_p = pearsonr(true_vals, pred_vals)
        spearman_r, spearman_p = spearmanr(true_vals, pred_vals)
        
        # Store correlations
        correlations[marker_name] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        }
        
        # Create scatter plot
        ax.scatter(true_vals, pred_vals, s=1, alpha=0.3, c='#3498DB')
        
        # Add diagonal reference line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, alpha=0.7, label='Identity')
        
        # Add regression line
        z = np.polyfit(true_vals, pred_vals, 1)
        p = np.poly1d(z)
        ax.plot(true_vals, p(true_vals), 'g-', linewidth=2, alpha=0.7, label='Regression')
        
        # Add statistics
        stats_text = f'Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_r:.3f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('True ADT Value')
        ax.set_ylabel('Predicted ADT Value')
        ax.set_title(f'{marker_name}')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_markers, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('ADT Marker Predictions: True vs. Predicted', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    
    plt.show()
    
    # Print summary
    print("\n📊 Correlation Summary:")
    for marker, stats in correlations.items():
        print(f"  {marker}: Pearson r={stats['pearson_r']:.3f}, Spearman ρ={stats['spearman_r']:.3f}")
    
    return fig, correlations


def plot_gene_protein_relationships(fused_embeddings, adt_true, adt_pred,
                                   marker_names=None, markers_to_plot=None,
                                   save_path='gene_protein_relationships.pdf'):
    """
    Plot relationship between gene embeddings and protein (ADT) values.
    
    Args:
        fused_embeddings: [N_cells, embedding_dim] - Fused embeddings
        adt_true: [N_cells, N_markers] - True ADT values
        adt_pred: [N_cells, N_markers] - Predicted ADT values
        marker_names: List of marker names
        markers_to_plot: List of specific markers to plot
        save_path: Path to save figure
    
    Returns:
        fig: Matplotlib figure object
    """
    print(f"📊 Generating gene-protein relationship plots...")
    
    # Convert to numpy if tensor
    if torch.is_tensor(fused_embeddings):
        fused_embeddings = fused_embeddings.cpu().numpy()
    if torch.is_tensor(adt_true):
        adt_true = adt_true.cpu().numpy()
    if torch.is_tensor(adt_pred):
        adt_pred = adt_pred.cpu().numpy()
    
    # Default markers to plot
    if markers_to_plot is None:
        markers_to_plot = ['CD19', 'CD3', 'CD34', 'cMPO', 'CD45', 'CD79a', 'CD7']
    
    # Find marker indices
    marker_indices = []
    marker_names_found = []
    
    if marker_names is not None:
        for marker in markers_to_plot:
            for i, name in enumerate(marker_names):
                if marker.lower() in name.lower():
                    marker_indices.append(i)
                    marker_names_found.append(name)
                    break
    else:
        marker_indices = list(range(min(7, adt_true.shape[1])))
        marker_names_found = [f'Marker_{i}' for i in marker_indices]
    
    n_markers = len(marker_indices)
    if n_markers == 0:
        print("⚠️ No matching markers found!")
        return None
    
    # Use first principal component of embeddings as gene expression proxy
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    embedding_pc1 = pca.fit_transform(fused_embeddings).flatten()
    
    # Create figure
    ncols = 4
    nrows = int(np.ceil(n_markers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = axes.flatten() if n_markers > 1 else [axes]
    
    for idx, (marker_idx, marker_name) in enumerate(zip(marker_indices, marker_names_found)):
        ax = axes[idx]
        
        # Get data for this marker
        protein_vals = adt_true[:, marker_idx]
        
        # Create hexbin plot for better visualization of dense data
        hb = ax.hexbin(embedding_pc1, protein_vals, gridsize=50, 
                      cmap='viridis', mincnt=1, alpha=0.8)
        
        # Add regression line
        z = np.polyfit(embedding_pc1, protein_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(embedding_pc1.min(), embedding_pc1.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.8, label='Trend')
        
        # Calculate correlation
        corr, _ = pearsonr(embedding_pc1, protein_vals)
        
        ax.set_xlabel('Gene Embedding (PC1)')
        ax.set_ylabel(f'{marker_name} Protein Level')
        ax.set_title(f'{marker_name} (r={corr:.3f})')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(hb, ax=ax, label='Cell Density')
    
    # Hide unused subplots
    for idx in range(n_markers, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Gene-Protein Relationships', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_aml_confusion_matrix(aml_true, aml_pred, 
                              class_names=None,
                              save_path='aml_confusion_matrix.pdf'):
    """
    Plot confusion matrix for AML vs Healthy classification.
    
    Args:
        aml_true: [N_cells] - True AML labels (0=Healthy, 1=AML)
        aml_pred: [N_cells] - Predicted AML probabilities or labels
        class_names: List of class names (default: ['Healthy', 'AML'])
        save_path: Path to save figure
    
    Returns:
        fig: Matplotlib figure object
        cm: Confusion matrix
        report: Classification report
    """
    print(f"📊 Generating AML classification confusion matrix...")
    
    # Convert to numpy if tensor
    if torch.is_tensor(aml_true):
        aml_true = aml_true.cpu().numpy()
    if torch.is_tensor(aml_pred):
        aml_pred = aml_pred.cpu().numpy()
    
    # Convert probabilities to binary labels if needed
    if aml_pred.ndim > 1:
        aml_pred = aml_pred.flatten()
    
    # Convert to binary (threshold at 0.5 if continuous)
    if aml_pred.dtype == np.float32 or aml_pred.dtype == np.float64:
        aml_pred_binary = (aml_pred > 0.5).astype(int)
    else:
        aml_pred_binary = aml_pred.astype(int)
    
    # Ensure true labels are int
    aml_true_binary = aml_true.astype(int)
    
    # Default class names
    if class_names is None:
        class_names = ['Healthy', 'AML']
    
    # Compute confusion matrix
    cm = confusion_matrix(aml_true_binary, aml_pred_binary)
    
    # Compute classification report
    report = classification_report(aml_true_binary, aml_pred_binary, 
                                  target_names=class_names, output_dict=True)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Confusion matrix with counts
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # Plot 2: Confusion matrix with percentages
    ax2 = axes[1]
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens',
               xticklabels=class_names, yticklabels=class_names,
               ax=ax2, cbar_kws={'label': 'Percentage (%)'})
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('Confusion Matrix (Percentages)')
    
    plt.suptitle('AML vs Healthy Classification Performance', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    
    plt.show()
    
    # Print classification report
    print("\n📊 Classification Report:")
    print(f"  Accuracy: {report['accuracy']:.3f}")
    for class_name in class_names:
        print(f"  {class_name}:")
        print(f"    Precision: {report[class_name]['precision']:.3f}")
        print(f"    Recall: {report[class_name]['recall']:.3f}")
        print(f"    F1-score: {report[class_name]['f1-score']:.3f}")
    
    return fig, cm, report


def create_comprehensive_visualization_report(model, rna_data, adt_data,
                                             rna_embeddings=None, adt_embeddings=None,
                                             fused_embeddings=None,
                                             adt_true=None, adt_pred=None,
                                             aml_true=None, aml_pred=None,
                                             cell_labels=None, cell_types=None,
                                             marker_names=None,
                                             save_dir='visualization_outputs/'):
    """
    Create comprehensive visualization report with all standard plots.
    
    Args:
        model: Trained GAT model
        rna_data: RNA PyG data
        adt_data: ADT PyG data
        rna_embeddings: RNA embeddings (optional, will be computed if None)
        adt_embeddings: ADT embeddings (optional, will be computed if None)
        fused_embeddings: Fused embeddings (optional, will be computed if None)
        adt_true: True ADT values
        adt_pred: Predicted ADT values
        aml_true: True AML labels
        aml_pred: Predicted AML probabilities
        cell_labels: Disease labels (AML vs Healthy)
        cell_types: Cell type annotations
        marker_names: ADT marker names
        save_dir: Directory to save all outputs
    
    Returns:
        dict: Dictionary with all generated figures and statistics
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("📊 GENERATING COMPREHENSIVE VISUALIZATION REPORT")
    print("="*60)
    
    results = {}
    
    # Extract embeddings if not provided
    if rna_embeddings is None or adt_embeddings is None or fused_embeddings is None:
        print("\n🔄 Extracting embeddings from model...")
        model.eval()
        with torch.no_grad():
            if adt_pred is None or aml_pred is None or fused_embeddings is None:
                adt_pred, aml_pred, fused_embeddings = model(
                    x=rna_data.x,
                    edge_index_rna=rna_data.edge_index,
                    edge_index_adt=adt_data.edge_index if hasattr(adt_data, 'edge_index') else None
                )
    
    # 1. UMAP Visualizations
    print("\n1️⃣ Generating UMAP visualizations...")
    
    if rna_embeddings is not None:
        fig_rna, umap_rna = plot_umap_rna_embeddings(
            rna_embeddings, cell_labels, cell_types,
            save_path=f'{save_dir}/umap_rna_embeddings.pdf'
        )
        results['umap_rna'] = {'fig': fig_rna, 'coords': umap_rna}
    
    if adt_embeddings is not None:
        fig_adt, umap_adt = plot_umap_adt_embeddings(
            adt_embeddings, cell_labels, cell_types,
            save_path=f'{save_dir}/umap_adt_embeddings.pdf'
        )
        results['umap_adt'] = {'fig': fig_adt, 'coords': umap_adt}
    
    if fused_embeddings is not None:
        fig_fused, umap_fused = plot_umap_fused_embeddings(
            fused_embeddings, cell_labels, cell_types,
            save_path=f'{save_dir}/umap_fused_embeddings.pdf'
        )
        results['umap_fused'] = {'fig': fig_fused, 'coords': umap_fused}
    
    # 2. ADT Marker Correlations
    if adt_true is not None and adt_pred is not None:
        print("\n2️⃣ Generating ADT marker correlation plots...")
        fig_corr, correlations = plot_adt_marker_correlations(
            adt_true, adt_pred, marker_names,
            save_path=f'{save_dir}/adt_marker_correlations.pdf'
        )
        results['adt_correlations'] = {'fig': fig_corr, 'stats': correlations}
    
    # 3. Gene-Protein Relationships
    if fused_embeddings is not None and adt_true is not None:
        print("\n3️⃣ Generating gene-protein relationship plots...")
        fig_gp = plot_gene_protein_relationships(
            fused_embeddings, adt_true, adt_pred, marker_names,
            save_path=f'{save_dir}/gene_protein_relationships.pdf'
        )
        results['gene_protein'] = {'fig': fig_gp}
    
    # 4. AML Confusion Matrix
    if aml_true is not None and aml_pred is not None:
        print("\n4️⃣ Generating AML confusion matrix...")
        fig_cm, cm, report = plot_aml_confusion_matrix(
            aml_true, aml_pred,
            save_path=f'{save_dir}/aml_confusion_matrix.pdf'
        )
        results['aml_classification'] = {'fig': fig_cm, 'cm': cm, 'report': report}
    
    print("\n" + "="*60)
    print("✅ COMPREHENSIVE VISUALIZATION REPORT COMPLETE!")
    print(f"📁 All outputs saved to: {save_dir}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    print("Standardized Visualization Module for Single-Cell Multi-Omics GAT Analysis")
    print("This module provides centralized visualization functions.")
    print("\nAvailable functions:")
    print("  - plot_umap_rna_embeddings()")
    print("  - plot_umap_adt_embeddings()")
    print("  - plot_umap_fused_embeddings()")
    print("  - plot_adt_marker_correlations()")
    print("  - plot_gene_protein_relationships()")
    print("  - plot_aml_confusion_matrix()")
    print("  - create_comprehensive_visualization_report()")

