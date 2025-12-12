"""Training script for quantum drug-protein interaction model."""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader
from torch_geometric.data import Batch
import warnings

warnings.filterwarnings("ignore")

from src.data.datasets import (
    load_pdb_ids_and_metadata,
    load_drugs,
    load_ligand_lookup,
    create_positive_pairs,
    DrugProteinDataset
)
from src.data.drug_graph import smiles_to_graph
from src.data.protein_graph import compute_contact_graph
from src.models.head import DrugProteinInteractionModel, ClassicalOnlyModel


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def seed_everything(seed: int):
    """Set all random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DrugProteinDataModule:
    """Data loading and preprocessing."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_config = config['data']
        self.pdb_dir = Path(self.data_config['pdb_dir'])
        
        # Load metadata
        self.pdb_metadata, self.excluded_ids = load_pdb_ids_and_metadata(
            self.data_config['include_glob'],
            self.data_config['exclude_glob']
        )
        
        self.protein_ids = sorted(list(self.pdb_metadata.keys()))
        
        # Load drugs
        self.drug_smiles = load_drugs(self.data_config['drugs_csv'])
        
        # Load optional ligand lookup
        self.ligand_lookup = load_ligand_lookup(
            self.data_config.get('ligand_lookup_csv')
        )
        
        # Create dataset
        positive_pairs = create_positive_pairs(
            self.protein_ids[:10],  # Limit to avoid long preprocessing
            self.pdb_metadata,
            self.ligand_lookup,
            str(self.pdb_dir)
        ) if self.ligand_lookup else None
        
        self.dataset = DrugProteinDataset(
            self.drug_smiles,
            self.protein_ids,
            self.pdb_metadata,
            positive_pairs=positive_pairs,
            negatives_per_positive=config['sampling']['negatives_per_positive']
        )
        
        # Split
        self.train_pairs, self.val_pairs, self.test_pairs = self.dataset.split(
            val_split=config['sampling']['val_split'],
            test_split=config['sampling']['test_split'],
            seed=config['sampling']['scaffold_seed'],
            split_by='protein'
        )
    
    def get_graphs(self, drug_id: str, protein_id: str) -> Tuple[Optional[object], Optional[object]]:
        """Get drug and protein graphs for a pair."""
        smiles = self.drug_smiles.get(drug_id)
        if not smiles:
            return None, None
        
        drug_graph = smiles_to_graph(smiles)
        if drug_graph is None:
            return None, None
        
        pdb_path = self.pdb_dir / f"{protein_id.lower()}.pdb"
        if not pdb_path.exists():
            return None, None
        
        protein_graph, _ = compute_contact_graph(str(pdb_path))
        
        return drug_graph, protein_graph
    
    def get_dataloaders(self, batch_size: int = 64):
        """Create train/val/test dataloaders."""
        
        def collate_fn(batch):
            """Collate batch of (drug_graph, protein_graph, label) tuples."""
            drug_graphs = []
            protein_graphs = []
            labels = []
            
            for drug_graph, protein_graph, label in batch:
                if drug_graph is not None and protein_graph is not None:
                    drug_graphs.append(drug_graph)
                    protein_graphs.append(protein_graph)
                    labels.append(label)
            
            if not drug_graphs:
                return None
            
            drug_batch = Batch.from_data_list(drug_graphs)
            protein_batch = Batch.from_data_list(protein_graphs)
            labels = torch.tensor(labels, dtype=torch.float32)
            
            return drug_batch, protein_batch, labels
        
        # Convert pair tuples to graph tuples
        def pairs_to_graphs(pairs):
            graph_data = []
            for drug_id, protein_id, binding_site, label in pairs:
                drug_graph, protein_graph = self.get_graphs(drug_id, protein_id)
                if drug_graph is not None and protein_graph is not None:
                    graph_data.append((drug_graph, protein_graph, label))
            return graph_data
        
        train_graphs = pairs_to_graphs(self.train_pairs)
        val_graphs = pairs_to_graphs(self.val_pairs)
        test_graphs = pairs_to_graphs(self.test_pairs)
        
        print(f"\nDataLoader sizes:")
        print(f"  Train: {len(train_graphs)}")
        print(f"  Val: {len(val_graphs)}")
        print(f"  Test: {len(test_graphs)}")
        
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: str = 'cpu'
) -> float:
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in train_loader:
        if batch is None:
            continue
        
        drug_batch, protein_batch, labels = batch
        drug_batch = drug_batch.to(device)
        protein_batch = protein_batch.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        scores, _, _ = model(drug_batch, protein_batch)
        loss = loss_fn(scores, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: str = 'cpu'
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Validate."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            
            drug_batch, protein_batch, labels = batch
            drug_batch = drug_batch.to(device)
            protein_batch = protein_batch.to(device)
            labels = labels.to(device)
            
            scores, _, _ = model(drug_batch, protein_batch)
            loss = loss_fn(scores, labels)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.extend(scores.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    
    avg_loss = total_loss / max(n_batches, 1)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return avg_loss, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--model_type', type=str, default='full', choices=['full', 'classical'])
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    seed_everything(config['seed'])
    
    # Device
    device = 'cpu'  # Force CPU for reproducibility
    print(f"Using device: {device}")
    
    # Create output dirs
    output_dir = Path(config['logging']['save_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Data
    print("\nLoading data...")
    dm = DrugProteinDataModule(config)
    train_loader, val_loader, test_loader = dm.get_dataloaders(
        batch_size=config['train']['batch_size']
    )
    
    if not train_loader or len(train_loader) == 0:
        print("ERROR: Empty train loader!")
        return
    
    # Model
    print("\nBuilding model...")
    if args.model_type == 'full':
        model = DrugProteinInteractionModel(config)
    else:
        model = ClassicalOnlyModel(config)
    
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train'].get('weight_decay', 0)
    )
    
    # Loss
    loss_fn = nn.BCELoss()
    
    # Training loop
    print(f"\nTraining ({args.model_type} model)...")
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(config['train']['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_preds, val_labels = validate(model, val_loader, loss_fn, device)
        
        print(f"Epoch {epoch+1}/{config['train']['epochs']}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save checkpoint
            checkpoint_path = output_dir / f"model_{args.model_type}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_loss, test_preds, test_labels = validate(model, test_loader, loss_fn, device)
    print(f"Test loss: {test_loss:.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        'predictions': test_preds,
        'labels': test_labels
    })
    results_df.to_csv(output_dir / f'test_results_{args.model_type}.csv', index=False)
    print(f"Saved results to {output_dir / f'test_results_{args.model_type}.csv'}")


if __name__ == '__main__':
    main()
