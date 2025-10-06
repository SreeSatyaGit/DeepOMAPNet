"""
finetune_rna_gat.py - Fine-tune RNA GAT model on new datasets

This script provides functionality to fine-tune a pre-trained RNA GAT model
on new datasets, with options for different fine-tuning strategies.

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
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
from data_provider.graph_data_builder import build_pyg_data
from trainer.adt_predictor import ADTPredictor

class RNAGATFineTuner:
    """
    Fine-tune RNA GAT model on new datasets.
    """

    def __init__(self, pretrained_model_path=None, individual_models_dir=None, device=None):
        """
        Initialize fine-tuner with pre-trained model.

        Args:
            pretrained_model_path (str): Path to pre-trained RNA GAT model
            individual_models_dir (str): Directory with individual model files
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.pretrained_model = None
        self.model_config = None
        self._load_pretrained_model(pretrained_model_path, individual_models_dir)

    def _load_pretrained_model(self, pretrained_model_path=None, individual_models_dir=None):
        """Load pre-trained RNA GAT model."""
        if individual_models_dir and os.path.exists(individual_models_dir):
            rna_gat_path = os.path.join(individual_models_dir, 'rna_gat_model.pth')
            if os.path.exists(rna_gat_path):
                checkpoint = torch.load(rna_gat_path, map_location=self.device)
                self.model_config = checkpoint['model_config']
                self.pretrained_model = SimpleGAT(**self.model_config)
                self.pretrained_model.load_state_dict(checkpoint['state_dict'])
                print(f"✅ Loaded pre-trained RNA GAT from: {rna_gat_path}")
            else:
                raise FileNotFoundError(f"RNA GAT model not found at: {rna_gat_path}")
        elif pretrained_model_path and os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location=self.device)
            if 'model_config' in checkpoint:
                self.model_config = checkpoint['model_config']
                self.pretrained_model = SimpleGAT(**self.model_config)
                self.pretrained_model.load_state_dict(checkpoint['state_dict'])
            else:
                self.pretrained_model = SimpleGAT(**self.model_config)
                self.pretrained_model.load_state_dict(checkpoint)
            print(f"✅ Loaded pre-trained RNA GAT from: {pretrained_model_path}")
        else:
            raise ValueError("Must provide either pretrained_model_path or individual_models_dir")

    def _preprocess_data(self, adata, use_existing_embeddings=True):
        """
        Preprocess RNA data for fine-tuning.

        Args:
            adata (AnnData): Input RNA data
            use_existing_embeddings (bool): Whether to use existing embeddings

        Returns:
            tuple: (processed_adata, pyg_data, use_rep)
        """
        print("Preprocessing data for fine-tuning...")

        adata_processed = adata.copy()

        if use_existing_embeddings and 'X_rna_embeddings' in adata_processed.obsm:
            print("Using existing RNA embeddings")
            use_rep = 'X_rna_embeddings'
        else:
            print("Running standard preprocessing pipeline...")

            sc.pp.filter_cells(adata_processed, min_genes=200)
            sc.pp.filter_genes(adata_processed, min_cells=3)

            adata_processed.raw = adata_processed
            sc.pp.normalize_total(adata_processed, target_sum=1e4)
            sc.pp.log1p(adata_processed)

            sc.pp.highly_variable_genes(adata_processed, min_mean=0.0125, max_mean=3, min_disp=0.5)
            adata_processed = adata_processed[:, adata_processed.var.highly_variable].copy()

            sc.pp.scale(adata_processed, max_value=10)
            sc.pp.pca(adata_processed, n_comps=50, svd_solver="arpack")
            use_rep = 'X_pca'

        sc.pp.neighbors(adata_processed, n_neighbors=15, use_rep=use_rep)

        pyg_data = build_pyg_data(adata_processed, use_pca=(use_rep == 'X_pca'))
        pyg_data = pyg_data.to(self.device)

        return adata_processed, pyg_data, use_rep

    def fine_tune_unsupervised(self, adata, epochs=100, learning_rate=0.001,
                             weight_decay=1e-4, batch_size=None, save_path=None):
        """
        Fine-tune RNA GAT model using unsupervised learning (reconstruction loss).

        Args:
            adata (AnnData): Input RNA data
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay
            batch_size (int): Batch size (if None, uses all data)
            save_path (str): Path to save fine-tuned model

        Returns:
            SimpleGAT: Fine-tuned model
        """
        print(f"Starting unsupervised fine-tuning for {epochs} epochs...")

        adata_processed, pyg_data, use_rep = self._preprocess_data(adata)

        fine_tuned_model = SimpleGAT(**self.model_config)
        fine_tuned_model.load_state_dict(self.pretrained_model.state_dict())
        fine_tuned_model = fine_tuned_model.to(self.device)

        fine_tuned_model.train()
        optimizer = optim.Adam(fine_tuned_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            if batch_size is None or batch_size >= pyg_data.x.shape[0]:
                embeddings = fine_tuned_model.get_embeddings(pyg_data.x, pyg_data.edge_index)
                loss = torch.mean(torch.norm(embeddings, dim=1))
            else:
                total_loss = 0
                num_batches = 0

                for i in range(0, pyg_data.x.shape[0], batch_size):
                    batch_x = pyg_data.x[i:i+batch_size]
                    batch_edge_index = pyg_data.edge_index

                    embeddings = fine_tuned_model.get_embeddings(batch_x, batch_edge_index)
                    loss = torch.mean(torch.norm(embeddings, dim=1))
                    total_loss += loss
                    num_batches += 1

                loss = total_loss / num_batches

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'state_dict': fine_tuned_model.state_dict(),
                'model_config': self.model_config,
                'fine_tuning_info': {
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'train_losses': train_losses,
                    'fine_tuning_timestamp': datetime.now().isoformat()
                }
            }, save_path)
            print(f"✅ Fine-tuned model saved to: {save_path}")

        return fine_tuned_model

    def fine_tune_supervised(self, adata, target_labels, epochs=100, learning_rate=0.001,
                           weight_decay=1e-4, batch_size=None, save_path=None):
        """
        Fine-tune RNA GAT model using supervised learning with target labels.

        Args:
            adata (AnnData): Input RNA data
            target_labels (array-like): Target labels for supervised learning
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay
            batch_size (int): Batch size
            save_path (str): Path to save fine-tuned model

        Returns:
            SimpleGAT: Fine-tuned model
        """
        print(f"Starting supervised fine-tuning for {epochs} epochs...")

        adata_processed, pyg_data, use_rep = self._preprocess_data(adata)

        if isinstance(target_labels, (list, np.ndarray)):
            target_labels = torch.tensor(target_labels, dtype=torch.long, device=self.device)

        fine_tuned_model = SimpleGAT(**self.model_config)
        fine_tuned_model.load_state_dict(self.pretrained_model.state_dict())
        fine_tuned_model = fine_tuned_model.to(self.device)

        fine_tuned_model.train()
        optimizer = optim.Adam(fine_tuned_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        train_losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            if batch_size is None or batch_size >= pyg_data.x.shape[0]:
                embeddings = fine_tuned_model.get_embeddings(pyg_data.x, pyg_data.edge_index)
                logits = torch.mean(embeddings, dim=1, keepdim=True).expand(-1, self.model_config['out_channels'])
                loss = criterion(logits, target_labels)
            else:
                total_loss = 0
                num_batches = 0

                for i in range(0, pyg_data.x.shape[0], batch_size):
                    batch_x = pyg_data.x[i:i+batch_size]
                    batch_labels = target_labels[i:i+batch_size]
                    batch_edge_index = pyg_data.edge_index

                    embeddings = fine_tuned_model.get_embeddings(batch_x, batch_edge_index)
                    logits = torch.mean(embeddings, dim=1, keepdim=True).expand(-1, self.model_config['out_channels'])
                    loss = criterion(logits, batch_labels)

                    total_loss += loss
                    num_batches += 1

                loss = total_loss / num_batches

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'state_dict': fine_tuned_model.state_dict(),
                'model_config': self.model_config,
                'fine_tuning_info': {
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'train_losses': train_losses,
                    'fine_tuning_timestamp': datetime.now().isoformat(),
                    'supervised': True
                }
            }, save_path)
            print(f"✅ Fine-tuned model saved to: {save_path}")

        return fine_tuned_model

    def fine_tune_contrastive(self, adata, epochs=100, learning_rate=0.001,
                            weight_decay=1e-4, temperature=0.1, save_path=None):
        """
        Fine-tune RNA GAT model using contrastive learning.

        Args:
            adata (AnnData): Input RNA data
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay
            temperature (float): Temperature for contrastive loss
            save_path (str): Path to save fine-tuned model

        Returns:
            SimpleGAT: Fine-tuned model
        """
        print(f"Starting contrastive fine-tuning for {epochs} epochs...")

        adata_processed, pyg_data, use_rep = self._preprocess_data(adata)

        fine_tuned_model = SimpleGAT(**self.model_config)
        fine_tuned_model.load_state_dict(self.pretrained_model.state_dict())
        fine_tuned_model = fine_tuned_model.to(self.device)

        fine_tuned_model.train()
        optimizer = optim.Adam(fine_tuned_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            embeddings = fine_tuned_model.get_embeddings(pyg_data.x, pyg_data.edge_index)

            embeddings_norm = torch.nn.functional.normalize(embeddings, dim=1)

            similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t()) / temperature

            loss = -torch.mean(torch.diag(similarity_matrix))

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'state_dict': fine_tuned_model.state_dict(),
                'model_config': self.model_config,
                'fine_tuning_info': {
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'temperature': temperature,
                    'train_losses': train_losses,
                    'fine_tuning_timestamp': datetime.now().isoformat(),
                    'contrastive': True
                }
            }, save_path)
            print(f"✅ Fine-tuned model saved to: {save_path}")

        return fine_tuned_model

    def evaluate_model(self, adata, model, target_labels=None):
        """
        Evaluate fine-tuned model on data.

        Args:
            adata (AnnData): Input RNA data
            model (SimpleGAT): Fine-tuned model
            target_labels (array-like): Target labels for evaluation

        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating fine-tuned model...")

        adata_processed, pyg_data, use_rep = self._preprocess_data(adata)

        model.eval()

        with torch.no_grad():
            embeddings = model.get_embeddings(pyg_data.x, pyg_data.edge_index)
            embeddings_np = embeddings.cpu().numpy()

        adata_processed.obsm['X_finetuned_embeddings'] = embeddings_np

        metrics = {
            'embedding_shape': embeddings_np.shape,
            'embedding_mean': np.mean(embeddings_np),
            'embedding_std': np.std(embeddings_np),
            'embedding_min': np.min(embeddings_np),
            'embedding_max': np.max(embeddings_np)
        }

        if target_labels is not None:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

            from sklearn.cluster import KMeans
            n_clusters = len(np.unique(target_labels))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            predicted_labels = kmeans.fit_predict(embeddings_np)

            metrics.update({
                'ari': adjusted_rand_score(target_labels, predicted_labels),
                'nmi': normalized_mutual_info_score(target_labels, predicted_labels)
            })

        print(f"✅ Evaluation complete!")
        for key, value in metrics.items():
            print(f"   {key}: {value}")

        return metrics, adata_processed

def fine_tune_rna_gat(adata, pretrained_model_path=None, individual_models_dir=None,
                     method='unsupervised', target_labels=None, epochs=100,
                     learning_rate=0.001, weight_decay=1e-4, batch_size=None,
                     save_path=None, device=None):
    """
    Convenience function to fine-tune RNA GAT model.

    Args:
        adata (AnnData): Input RNA data
        pretrained_model_path (str): Path to pre-trained model
        individual_models_dir (str): Directory with individual model files
        method (str): Fine-tuning method ('unsupervised', 'supervised', 'contrastive')
        target_labels (array-like): Target labels for supervised learning
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay
        batch_size (int): Batch size
        save_path (str): Path to save fine-tuned model
        device (str): Device to use

    Returns:
        SimpleGAT: Fine-tuned model
    """
    fine_tuner = RNAGATFineTuner(
        pretrained_model_path=pretrained_model_path,
        individual_models_dir=individual_models_dir,
        device=device
    )

    if method == 'unsupervised':
        fine_tuned_model = fine_tuner.fine_tune_unsupervised(
            adata, epochs=epochs, learning_rate=learning_rate,
            weight_decay=weight_decay, batch_size=batch_size, save_path=save_path
        )
    elif method == 'supervised':
        if target_labels is None:
            raise ValueError("target_labels must be provided for supervised fine-tuning")
        fine_tuned_model = fine_tuner.fine_tune_supervised(
            adata, target_labels, epochs=epochs, learning_rate=learning_rate,
            weight_decay=weight_decay, batch_size=batch_size, save_path=save_path
        )
    elif method == 'contrastive':
        fine_tuned_model = fine_tuner.fine_tune_contrastive(
            adata, epochs=epochs, learning_rate=learning_rate,
            weight_decay=weight_decay, save_path=save_path
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return fine_tuned_model

if __name__ == "__main__":
    print("RNA GAT Fine-tuning Module")
    print("=" * 50)
    print("Usage:")
    print("  from scripts.finetune_rna_gat import fine_tune_rna_gat")
    print("  fine_tuned_model = fine_tune_rna_gat(")
    print("      adata, individual_models_dir='...', method='unsupervised')")
    print()
    print("Or using the class:")
    print("  from scripts.finetune_rna_gat import RNAGATFineTuner")
    print("  fine_tuner = RNAGATFineTuner(individual_models_dir='...')")
    print("  fine_tuned_model = fine_tuner.fine_tune_unsupervised(adata)")
