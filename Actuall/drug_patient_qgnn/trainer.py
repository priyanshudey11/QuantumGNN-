"""
Training module for Quantum Drug-Protein GNN.

This module provides:
- DrugProteinTrainer: Main training class with fit/evaluate methods
- Training loop with batching
- Metrics: loss, accuracy, AUC-ROC
- Train/validation split

Backward compatibility:
- DrugPatientTrainer is aliased to DrugProteinTrainer
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score
import time


class DrugProteinTrainer:
    """Trainer for drug-protein interaction prediction model.

    Handles:
    - Training loop with mini-batches
    - Train/validation split
    - Metric tracking (loss, accuracy, AUC-ROC)
    - Model checkpointing

    Args:
        model: QuantumDrugProteinGNN model instance
        learning_rate: Learning rate for optimizer (default: 0.001)
        device: Device to train on ('cpu', 'cuda', or 'mps')
        optimizer: Optional custom optimizer (default: Adam)
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        self.model = model.to(self.device)

        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )
        else:
            self.optimizer = optimizer

        # Loss function (binary cross-entropy)
        self.criterion = nn.BCELoss()

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': []
        }

    def _prepare_batch_data(
        self,
        graph,
        indices: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch data from graph edges.

        Args:
            graph: BipartiteGraph object
            indices: Indices of edges to include in batch

        Returns:
            Tuple of (ligand_features, pocket_features, labels)
        """
        # Get full feature matrices (use backward-compatible methods)
        ligand_feature_matrix = graph.get_drug_features_matrix()  # Uses backward-compatible method
        pocket_feature_matrix = graph.get_patient_features_matrix()  # Uses backward-compatible method
        edge_index, edge_features = graph.get_edge_index()

        # Extract batch edges
        batch_ligand_indices = edge_index[0, indices]
        batch_pocket_indices = edge_index[1, indices]

        # Get features for this batch
        batch_ligand_features = ligand_feature_matrix[batch_ligand_indices]
        batch_pocket_features = pocket_feature_matrix[batch_pocket_indices]

        # Get labels (outcome column from edge features - now index 0 instead of 3)
        batch_labels = edge_features[indices, 0]  # outcome is index 0 in new format

        # Convert to tensors
        ligand_features = torch.tensor(batch_ligand_features, dtype=torch.float32)
        pocket_features = torch.tensor(batch_pocket_features, dtype=torch.float32)
        labels = torch.tensor(batch_labels, dtype=torch.float32)

        return ligand_features, pocket_features, labels

    def train_epoch(
        self,
        graph,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            graph: BipartiteGraph object
            batch_size: Batch size for training
            shuffle: Whether to shuffle data each epoch

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        n_edges = graph.num_edges()
        indices = np.arange(n_edges)

        if shuffle:
            np.random.shuffle(indices)

        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Process batches
        n_batches = (n_edges + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_edges)
            batch_indices = indices[start_idx:end_idx]

            # Prepare batch
            ligand_features, pocket_features, labels = self._prepare_batch_data(
                graph, batch_indices
            )

            # Move to device
            ligand_features = ligand_features.to(self.device)
            pocket_features = pocket_features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(ligand_features, pocket_features).squeeze(-1)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * len(batch_indices)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate epoch metrics
        avg_loss = total_loss / n_edges
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = accuracy_score(all_labels, (all_preds >= 0.5).astype(int))

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def evaluate(
        self,
        graph,
        indices: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """Evaluate model on given data.

        Args:
            graph: BipartiteGraph object
            indices: Indices of edges to evaluate
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        n_samples = len(indices)
        n_batches = (n_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                # Prepare batch
                ligand_features, pocket_features, labels = self._prepare_batch_data(
                    graph, batch_indices
                )

                # Move to device
                ligand_features = ligand_features.to(self.device)
                pocket_features = pocket_features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(ligand_features, pocket_features).squeeze(-1)
                loss = self.criterion(outputs, labels)

                # Track metrics
                total_loss += loss.item() * len(batch_indices)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / n_samples
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = accuracy_score(all_labels, (all_preds >= 0.5).astype(int))

        # Calculate AUC-ROC if we have both classes
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
        else:
            auc = float('nan')

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc
        }

    def fit(
        self,
        graph,
        epochs: int = 100,
        batch_size: int = 32,
        val_split: float = 0.2,
        verbose: int = 1,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """Train the model on the graph.

        Args:
            graph: BipartiteGraph object
            epochs: Number of training epochs
            batch_size: Batch size for training
            val_split: Fraction of data to use for validation
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
            early_stopping_patience: Stop if validation loss doesn't improve for N epochs

        Returns:
            Training history dictionary
        """
        n_edges = graph.num_edges()

        if n_edges == 0:
            raise ValueError("Graph has no edges. Cannot train.")

        # Create train/val split
        indices = np.arange(n_edges)
        np.random.shuffle(indices)

        n_val = int(n_edges * val_split)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        if verbose > 0:
            print(f"\nTraining on {len(train_indices)} edges, validating on {len(val_indices)} edges")
            print(f"Model: {self.model.get_model_info()}")
            print(f"Device: {self.device}\n")

        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(graph, batch_size=batch_size, shuffle=True)

            # Validate
            if len(val_indices) > 0:
                val_metrics = self.evaluate(graph, val_indices, batch_size=batch_size)
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0, 'auc': 0.0}

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_auc'].append(val_metrics['auc'])

            epoch_time = time.time() - epoch_start

            # Print progress
            if verbose >= 2:
                print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - "
                      f"train_loss: {train_metrics['loss']:.4f} - "
                      f"train_acc: {train_metrics['accuracy']:.4f} - "
                      f"val_loss: {val_metrics['loss']:.4f} - "
                      f"val_acc: {val_metrics['accuracy']:.4f} - "
                      f"val_auc: {val_metrics['auc']:.4f}")
            elif verbose == 1 and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"val_loss: {val_metrics['loss']:.4f} - "
                      f"val_acc: {val_metrics['accuracy']:.4f} - "
                      f"val_auc: {val_metrics['auc']:.4f}")

            # Early stopping
            if early_stopping_patience is not None:
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose > 0:
                        print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break

        if verbose > 0:
            print("\nTraining complete!")
            print(f"Final metrics:")
            print(f"  Train Loss: {self.history['train_loss'][-1]:.4f}")
            print(f"  Train Acc:  {self.history['train_acc'][-1]:.4f}")
            print(f"  Val Loss:   {self.history['val_loss'][-1]:.4f}")
            print(f"  Val Acc:    {self.history['val_acc'][-1]:.4f}")
            print(f"  Val AUC:    {self.history['val_auc'][-1]:.4f}")

        return self.history

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_config': self.model.get_model_info()
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint.

        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {filepath}")


# Backward compatibility alias
DrugPatientTrainer = DrugProteinTrainer
