

import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.gat_models import SimpleGAT
from model.transformer_models import TransformerMapping
from data_provider.graph_data_builder import build_pyg_data

class ADTPredictor:
    """
    A class to predict ADT embeddings and cell types from RNA data using trained models.
    """

    def __init__(self, checkpoint_path=None, individual_models_dir=None, device=None):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)

        self.rna_gat_model, self.adt_gat_model, self.transformer_model, self.checkpoint_info = self._load_models(
            checkpoint_path, individual_models_dir
        )


    def _load_models(self, checkpoint_path=None, individual_models_dir=None):
        
        if individual_models_dir and os.path.exists(individual_models_dir.strip()):
            return self._load_individual_models(individual_models_dir.strip())

        if checkpoint_path is None:
            possible_paths = [
                # Latest models from your training
                "/projects/vanaja_lab/satya/DeepOMAPNet/Notebooks/trained_models/rna_adt_transformer_models_20251006_130642.pth",
                "/projects/vanaja_lab/satya/DeepOMAPNet/Notebooks/trained_models/rna_adt_transformer_models_20251006_130513.pth",
                "/projects/vanaja_lab/satya/DeepOMAPNet/Notebooks/trained_models/rna_adt_transformer_models_20251006_130438.pth",
                "/projects/vanaja_lab/satya/DeepOMAPNet/Notebooks/trained_models/individual_models_20251006_130642",
                "/projects/vanaja_lab/satya/DeepOMAPNet/Notebooks/trained_models/individual_models_20251006_130513",
                "/projects/vanaja_lab/satya/DeepOMAPNet/Notebooks/trained_models/individual_models_20251006_130438",
                # Legacy paths
                "trained_models/rna_adt_transformer_models_20250922_115253.pth",
                "rna_adt_transformer_mapping_models.pth",
                "../trained_models/rna_adt_transformer_models_20250922_115253.pth",
                "../rna_adt_transformer_mapping_models.pth",
                "Notebooks/trained_models/rna_adt_transformer_models_20250922_115253.pth",
                "Notebooks/trained_models/individual_models_20250922_115253"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    if os.path.isdir(path) and "individual_models" in path:
                        return self._load_individual_models(path)
                    elif path.endswith('.pth'):
                        checkpoint_path = path
                        break

            if checkpoint_path is None:
                raise FileNotFoundError(
                    "No checkpoint file found. Please ensure you have trained models. "
                    "Expected paths: trained_models/rna_adt_transformer_models_*.pth or "
                    "individual_models directory"
                )


        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        rna_input_dim = checkpoint['rna_input_dim']
        adt_output_dim = checkpoint['adt_output_dim']
        rna_num_classes = checkpoint['rna_num_classes']
        adt_num_classes = checkpoint['adt_num_classes']

        rna_gat_model = SimpleGAT(
            in_channels=rna_input_dim,
            hidden_channels=64,
            out_channels=rna_num_classes,
            heads=4,
            dropout=0.6
        ).to(self.device)

        adt_gat_model = SimpleGAT(
            in_channels=50,
            hidden_channels=64,
            out_channels=adt_num_classes,
            heads=4,
            dropout=0.6
        ).to(self.device)

        # Get the actual input dimension from the checkpoint
        transformer_input_dim = checkpoint['transformer_mapping_state_dict']['input_proj.weight'].shape[1]
        
        transformer_model = TransformerMapping(
            input_dim=transformer_input_dim,
            output_dim=adt_output_dim,
            d_model=256,
            nhead=4,
            num_layers=3
        ).to(self.device)

        rna_gat_model.load_state_dict(checkpoint['rna_gat_state_dict'])
        adt_gat_model.load_state_dict(checkpoint['adt_gat_state_dict'])
        transformer_model.load_state_dict(checkpoint['transformer_mapping_state_dict'])

        rna_gat_model.eval()
        adt_gat_model.eval()
        transformer_model.eval()

        checkpoint_info = {
            'rna_input_dim': rna_input_dim,
            'adt_output_dim': adt_output_dim,
            'rna_num_classes': rna_num_classes,
            'adt_num_classes': adt_num_classes,
            'checkpoint_path': checkpoint_path
        }

        return rna_gat_model, adt_gat_model, transformer_model, checkpoint_info

    def _load_individual_models(self, models_dir):
        

        rna_gat_path = os.path.join(models_dir, 'rna_gat_model.pth')
        if not os.path.exists(rna_gat_path):
            raise FileNotFoundError(f"RNA GAT model not found at: {rna_gat_path}")

        rna_gat_checkpoint = torch.load(rna_gat_path, map_location='cpu')
        rna_gat_model = SimpleGAT(**rna_gat_checkpoint['model_config']).to(self.device)
        rna_gat_model.load_state_dict(rna_gat_checkpoint['state_dict'])
        rna_gat_model.eval()

        adt_gat_path = os.path.join(models_dir, 'adt_gat_model.pth')
        if not os.path.exists(adt_gat_path):
            raise FileNotFoundError(f"ADT GAT model not found at: {adt_gat_path}")

        adt_gat_checkpoint = torch.load(adt_gat_path, map_location='cpu')
        adt_gat_model = SimpleGAT(**adt_gat_checkpoint['model_config']).to(self.device)
        adt_gat_model.load_state_dict(adt_gat_checkpoint['state_dict'])
        adt_gat_model.eval()

        transformer_path = os.path.join(models_dir, 'transformer_mapping_model.pth')
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(f"Transformer model not found at: {transformer_path}")

        transformer_checkpoint = torch.load(transformer_path, map_location='cpu')
        
        # Use the actual dimensions from the saved model config
        transformer_config = transformer_checkpoint['model_config'].copy()
        transformer_model = TransformerMapping(**transformer_config).to(self.device)
        transformer_model.load_state_dict(transformer_checkpoint['state_dict'])
        transformer_model.eval()

        rna_input_dim = rna_gat_checkpoint['model_config']['in_channels']
        adt_output_dim = transformer_checkpoint['model_config']['output_dim']
        rna_num_classes = rna_gat_checkpoint['model_config']['out_channels']
        adt_num_classes = adt_gat_checkpoint['model_config']['out_channels']

        checkpoint_info = {
            'rna_input_dim': rna_input_dim,
            'adt_output_dim': adt_output_dim,
            'rna_num_classes': rna_num_classes,
            'adt_num_classes': adt_num_classes,
            'checkpoint_path': models_dir,
            'model_type': 'individual_models'
        }

        return rna_gat_model, adt_gat_model, transformer_model, checkpoint_info

    def _preprocess_rna_data(self, adata, use_existing_embeddings=True):
        
        adata_processed = adata.copy()

        if 'X_integrated.cca' not in adata_processed.obsm or not use_existing_embeddings:
            sc.pp.normalize_total(adata_processed, target_sum=1e4)
            sc.pp.log1p(adata_processed)

        if 'X_integrated.cca' in adata_processed.obsm and use_existing_embeddings:
            use_rep = 'X_integrated.cca'
        else:
            
            # Handle NaN values and ensure data is valid
            if adata_processed.X is not None:
                # Check for NaN values
                if hasattr(adata_processed.X, 'data'):
                    # Sparse matrix
                    nan_count = np.isnan(adata_processed.X.data).sum()
                else:
                    # Dense matrix
                    nan_count = np.isnan(adata_processed.X).sum()
                
                if nan_count > 0:
                    pass
            
            # Try to find highly variable genes with error handling
            try:
                sc.pp.highly_variable_genes(adata_processed, n_top_genes=2000)
                adata_processed = adata_processed[:, adata_processed.var.highly_variable].copy()
            except Exception as e:
                pass
            
            # Scale the data
            sc.pp.scale(adata_processed, max_value=10)
            
            # Compute PCA
            sc.tl.pca(adata_processed, n_comps=50, svd_solver="arpack")
            use_rep = 'X_pca'

        sc.pp.neighbors(adata_processed, n_neighbors=15, use_rep=use_rep)

        pyg_data = build_pyg_data(adata_processed, use_pca=(use_rep == 'X_pca'))
        pyg_data = pyg_data.to(self.device)

        return adata_processed, pyg_data, use_rep

    def predict_adt_embeddings(self, adata, use_existing_embeddings=True, batch_size=None):
        
        adata_processed, pyg_data, use_rep = self._preprocess_rna_data(adata, use_existing_embeddings)

        with torch.no_grad():
            rna_embeddings = self.rna_gat_model.get_embeddings(pyg_data.x, pyg_data.edge_index)

        with torch.no_grad():
            try:
                if batch_size is None or batch_size >= rna_embeddings.shape[0]:
                    predicted_adt_embeddings = self.transformer_model(rna_embeddings)
                else:
                    predicted_adt_embeddings = []
                    for i in range(0, rna_embeddings.shape[0], batch_size):
                        batch = rna_embeddings[i:i+batch_size]
                        batch_pred = self.transformer_model(batch)
                        predicted_adt_embeddings.append(batch_pred)
                    predicted_adt_embeddings = torch.cat(predicted_adt_embeddings, dim=0)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    transformer_cpu = self.transformer_model.cpu()
                    rna_embeddings_cpu = rna_embeddings.cpu()
                    
                    if batch_size is None or batch_size >= rna_embeddings_cpu.shape[0]:
                        predicted_adt_embeddings = transformer_cpu(rna_embeddings_cpu)
                    else:
                        predicted_adt_embeddings = []
                        for i in range(0, rna_embeddings_cpu.shape[0], batch_size):
                            batch = rna_embeddings_cpu[i:i+batch_size]
                            batch_pred = transformer_cpu(batch)
                            predicted_adt_embeddings.append(batch_pred)
                        predicted_adt_embeddings = torch.cat(predicted_adt_embeddings, dim=0)
                    
                    predicted_adt_embeddings = predicted_adt_embeddings.to(self.device)
                else:
                    raise e

        rna_embeddings_np = rna_embeddings.cpu().numpy()
        predicted_adt_embeddings_np = predicted_adt_embeddings.cpu().numpy()

        return rna_embeddings_np, predicted_adt_embeddings_np

    def predict_cell_types(self, predicted_adt_embeddings, method='kmeans', n_clusters=None,
                          cell_type_names=None, reference_cell_types=None):
        
        if method == 'kmeans':
            from sklearn.cluster import KMeans

            if n_clusters is None:
                n_clusters = min(20, predicted_adt_embeddings.shape[0] // 100)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            predicted_labels = kmeans.fit_predict(predicted_adt_embeddings)

            distances = kmeans.transform(predicted_adt_embeddings)
            confidence_scores = 1.0 / (1.0 + distances.min(axis=1))

        elif method == 'leiden':
            temp_adata = ad.AnnData(X=predicted_adt_embeddings)
            sc.pp.neighbors(temp_adata, n_neighbors=15, use_rep='X')
            sc.tl.leiden(temp_adata, resolution=1.0)
            predicted_labels = temp_adata.obs['leiden'].astype(int).values

            cluster_counts = pd.Series(predicted_labels).value_counts()
            confidence_scores = cluster_counts[predicted_labels].values / len(predicted_labels)

        elif method == 'reference' and reference_cell_types is not None:
            from sklearn.neighbors import KNeighborsClassifier

            knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
            knn.fit(predicted_adt_embeddings, reference_cell_types)
            predicted_labels = knn.predict(predicted_adt_embeddings)

            neighbor_distances, neighbor_indices = knn.kneighbors(predicted_adt_embeddings)
            confidence_scores = 1.0 / (1.0 + neighbor_distances.mean(axis=1))

        else:
            raise ValueError(f"Unknown method: {method}")

        if cell_type_names is None:
            unique_labels = np.unique(predicted_labels)
            if method == 'reference' and reference_cell_types is not None:
                cell_type_names = [f"CellType_{label}" for label in unique_labels]
            else:
                cell_type_names = [f"Cluster_{label}" for label in unique_labels]

        label_to_name = {label: name for label, name in zip(unique_labels, cell_type_names)}
        predicted_cell_types = [label_to_name[label] for label in predicted_labels]

        return predicted_cell_types, cell_type_names, confidence_scores

    def _get_adt_marker_names(self):
        
        try:
            # Try to load one of the training ADT files to get marker names
            import scanpy as sc
            
            # Try different possible paths for ADT training data
            adt_paths = [
                "/projects/vanaja_lab/satya/Datasets/ControlADT.h5ad",
                "/projects/vanaja_lab/satya/Datasets/AMLAADT.h5ad", 
                "/projects/vanaja_lab/satya/Datasets/AMLBADT.h5ad"
            ]
            
            for adt_path in adt_paths:
                if os.path.exists(adt_path):
                    adt_data = sc.read_h5ad(adt_path)
                    marker_names = adt_data.var_names.tolist()
                    
                    expected_dim = self.checkpoint_info['adt_output_dim']
                    if len(marker_names) == expected_dim:
                        return marker_names
                    elif len(marker_names) > expected_dim:
                        return marker_names[:expected_dim]
                    else:
                        generic_names = [f'predicted_adt_{i}' for i in range(expected_dim)]
                        for i, name in enumerate(marker_names):
                            if i < expected_dim:
                                generic_names[i] = name
                        return generic_names
            
            return [f'predicted_adt_{i}' for i in range(self.checkpoint_info['adt_output_dim'])]
            
        except Exception as e:
            return [f'predicted_adt_{i}' for i in range(self.checkpoint_info['adt_output_dim'])]

    def add_predictions_to_adata(self, adata, use_existing_embeddings=True,
                                batch_size=None, save_embeddings=True,
                                adt_marker_names=None, predict_cell_types=True,
                                cell_type_method='kmeans', n_clusters=None,
                                cell_type_names=None, use_actual_marker_names=True):
        
        rna_embeddings_np, predicted_adt_embeddings_np = self.predict_adt_embeddings(
            adata, use_existing_embeddings, batch_size
        )

        adata_with_predictions = adata.copy()

        if adt_marker_names is None:
            if use_actual_marker_names:
                # Get actual ADT marker names from training data
                actual_marker_names = self._get_adt_marker_names()
                # Use actual names if available, otherwise use generic names
                if len(actual_marker_names) == predicted_adt_embeddings_np.shape[1]:
                    adt_marker_names = [f'predicted_{name}' for name in actual_marker_names]
                else:
                    adt_marker_names = [f'predicted_adt_{i}' for i in range(predicted_adt_embeddings_np.shape[1])]
            else:
                adt_marker_names = [f'predicted_adt_{i}' for i in range(predicted_adt_embeddings_np.shape[1])]

        for i, marker_name in enumerate(adt_marker_names):
            adata_with_predictions.obs[marker_name] = predicted_adt_embeddings_np[:, i]

        if predict_cell_types:
            predicted_cell_types, cell_type_names, confidence_scores = self.predict_cell_types(
                predicted_adt_embeddings_np,
                method=cell_type_method,
                n_clusters=n_clusters,
                cell_type_names=cell_type_names
            )

            adata_with_predictions.obs['predicted_cell_type'] = predicted_cell_types
            adata_with_predictions.obs['cell_type_confidence'] = confidence_scores

        if save_embeddings:
            adata_with_predictions.obsm['X_rna_embeddings'] = rna_embeddings_np
            adata_with_predictions.obsm['X_predicted_adt_embeddings'] = predicted_adt_embeddings_np
            adata_with_predictions.uns['predicted_adt_data'] = predicted_adt_embeddings_np

        metadata = {
            'model_checkpoint': self.checkpoint_info['checkpoint_path'],
            'rna_embedding_dim': rna_embeddings_np.shape[1],
            'adt_embedding_dim': predicted_adt_embeddings_np.shape[1],
            'prediction_timestamp': datetime.now().isoformat(),
            'adt_marker_names': adt_marker_names,
            'device_used': str(self.device)
        }

        if predict_cell_types:
            metadata.update({
                'cell_type_method': cell_type_method,
                'n_predicted_cell_types': len(np.unique(predicted_cell_types)),
                'cell_type_names': cell_type_names
            })

        adata_with_predictions.uns['prediction_info'] = metadata

        return adata_with_predictions

def predict_adt_from_rna(adata, checkpoint_path=None, individual_models_dir=None,
                        use_existing_embeddings=True, batch_size=None,
                        adt_marker_names=None, device=None, predict_cell_types=True,
                        cell_type_method='kmeans', n_clusters=None, cell_type_names=None,
                        use_actual_marker_names=True):
    
    predictor = ADTPredictor(
        checkpoint_path=checkpoint_path,
        individual_models_dir=individual_models_dir,
        device=device
    )

    adata_with_predictions = predictor.add_predictions_to_adata(
        adata=adata,
        use_existing_embeddings=use_existing_embeddings,
        batch_size=batch_size,
        adt_marker_names=adt_marker_names,
        predict_cell_types=predict_cell_types,
        cell_type_method=cell_type_method,
        n_clusters=n_clusters,
        cell_type_names=cell_type_names,
        use_actual_marker_names=use_actual_marker_names
    )

    return adata_with_predictions
