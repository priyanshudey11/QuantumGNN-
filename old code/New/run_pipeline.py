"""Standalone training script with full pipeline."""

import os
import sys
import yaml
import argparse
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from src.data.datasets import (
    load_pdb_ids_and_metadata,
    load_drugs,
    load_ligand_lookup,
    DrugProteinDataset,
    load_from_alternate_source
)
from src.data.enhanced_loader import (
    load_data_with_alternate_support,
    create_dataset_from_alternate_source
)
from src.data.drug_graph import smiles_to_graph
from src.data.protein_graph import compute_contact_graph
from src.models.head import DrugProteinInteractionModel, ClassicalOnlyModel
from src.evaluate import compute_metrics, bootstrap_metrics, MetricsReporter


def seed_everything(seed: int):
    """Set all random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Pipeline:
    """Complete training and evaluation pipeline."""
    
    def __init__(self, config_path: str, model_type: str = 'full'):
        self.config_path = config_path
        self.model_type = model_type
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        seed_everything(self.config['seed'])
        
        self.device = self.config.get('device', 'cpu')
        self.output_dir = Path(self.config['logging']['save_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Quantum Drug-Protein Interaction Pipeline")
        print(f"Model Type: {model_type}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
    
    def load_data(self):
        """Load and prepare data from standard or alternate source."""
        print("Loading data...")
        
        # Use enhanced loader for both standard and alternate paths
        self.pdb_metadata, self.drug_smiles, self.ligand_lookup = load_data_with_alternate_support(
            self.config
        )
        
        self.protein_ids = sorted(list(self.pdb_metadata.keys()))
        print(f" Loaded {len(self.protein_ids)} proteins")
        print(f" Loaded {len(self.drug_smiles)} drugs")

        # Create positive pairs from ligand_lookup
        positive_pairs = []
        if self.ligand_lookup:
            for lig_code, smiles in self.ligand_lookup.items():
                # Parse PDB ID from ligand code (format: pdbid_LIG)
                if '_' in lig_code:
                    pdb_id = lig_code.split('_')[0]
                    # Try both lowercase and uppercase versions
                    pdb_id_lower = pdb_id.lower()
                    pdb_id_upper = pdb_id.upper()

                    if pdb_id_lower in self.pdb_metadata:
                        binding_site = self.pdb_metadata[pdb_id_lower].get('binding_site', 'unknown')
                        positive_pairs.append((smiles, pdb_id_lower, binding_site))
                    elif pdb_id_upper in self.pdb_metadata:
                        binding_site = self.pdb_metadata[pdb_id_upper].get('binding_site', 'unknown')
                        positive_pairs.append((smiles, pdb_id_upper, binding_site))

        print(f" Created {len(positive_pairs)} positive pairs from ligand lookup")

        # Create dataset
        self.dataset = DrugProteinDataset(
            self.drug_smiles,
            self.protein_ids,
            self.pdb_metadata,
            positive_pairs=positive_pairs if positive_pairs else None,
            negatives_per_positive=self.config['sampling']['negatives_per_positive']
        )
        
        # Split
        self.train_pairs, self.val_pairs, self.test_pairs = self.dataset.split(
            val_split=self.config['sampling']['val_split'],
            test_split=self.config['sampling']['test_split'],
            seed=self.config['sampling']['scaffold_seed']
        )
        
        print(f" Train: {len(self.train_pairs)}, Val: {len(self.val_pairs)}, Test: {len(self.test_pairs)}")
    
    def get_graphs(self, drug_id: str, protein_id: str):
        """Load graphs for a pair."""
        pdb_dir = Path(self.config['data']['pdb_dir'])
        smiles = self.drug_smiles.get(drug_id)
        
        if not smiles:
            return None, None
        
        drug_graph = smiles_to_graph(smiles)
        if drug_graph is None:
            return None, None
        
        pdb_path = pdb_dir / f"{protein_id.lower()}.pdb"
        if not pdb_path.exists():
            return None, None
        
        protein_graph, _ = compute_contact_graph(str(pdb_path))
        return drug_graph, protein_graph
    
    def create_dataloaders(self):
        """Create train/val/test dataloaders."""
        print("\nPreparing dataloaders...")
        
        def pairs_to_graphs(pairs):
            graphs = []
            for drug_id, protein_id, binding_site, label in pairs:
                drug_g, prot_g = self.get_graphs(drug_id, protein_id)
                if drug_g and prot_g:
                    graphs.append((drug_g, prot_g, label))
            return graphs
        
        def collate_fn(batch):
            if not batch:
                return None
            drug_graphs, prot_graphs, labels = zip(*batch)
            drug_batch = Batch.from_data_list(list(drug_graphs))
            prot_batch = Batch.from_data_list(list(prot_graphs))
            return drug_batch, prot_batch, torch.tensor(labels, dtype=torch.float32)
        
        train_graphs = pairs_to_graphs(self.train_pairs)
        val_graphs = pairs_to_graphs(self.val_pairs)
        test_graphs = pairs_to_graphs(self.test_pairs)
        
        print(f"  Train graphs: {len(train_graphs)}")
        print(f"  Val graphs: {len(val_graphs)}")
        print(f"  Test graphs: {len(test_graphs)}")
        
        batch_size = self.config['train']['batch_size']
        
        self.train_loader = DataLoader(
            train_graphs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            val_graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        self.test_loader = DataLoader(
            test_graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
    
    def build_model(self):
        """Build model."""
        print("\nBuilding model...")
        
        if self.model_type == 'full':
            self.model = DrugProteinInteractionModel(self.config)
        else:
            self.model = ClassicalOnlyModel(self.config)
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f" Model created ({n_params:,} parameters)")
    
    def train_epoch(self):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in self.train_loader:
            if batch is None:
                continue
            
            drug_batch, prot_batch, labels = batch
            drug_batch = drug_batch.to(self.device)
            prot_batch = prot_batch.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            scores, _, _ = self.model(drug_batch, prot_batch)
            loss = self.loss_fn(scores, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def validate(self, loader):
        """Validate."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        n_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                
                drug_batch, prot_batch, labels = batch
                drug_batch = drug_batch.to(self.device)
                prot_batch = prot_batch.to(self.device)
                labels = labels.to(self.device)
                
                scores, _, _ = self.model(drug_batch, prot_batch)
                loss = self.loss_fn(scores, labels)
                
                total_loss += loss.item()
                n_batches += 1
                
                all_preds.extend(scores.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
        
        avg_loss = total_loss / max(n_batches, 1)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        return avg_loss, all_preds, all_labels
    
    def train(self):
        """Main training loop."""
        print("\nTraining...")

        # Optimizer (convert lr and weight_decay to float in case YAML parses them as strings)
        lr = float(self.config['train']['lr'])
        weight_decay = float(self.config['train'].get('weight_decay', 0))

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Loss
        self.loss_fn = nn.BCELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.config['train']['epochs']):
            train_loss = self.train_epoch()
            val_loss, val_preds, val_labels = self.validate(self.val_loader)
            
            # Metrics
            if len(np.unique(val_labels)) > 1:
                val_auroc = compute_metrics(val_labels, val_preds)['auroc']
            else:
                val_auroc = 0.0
            
            print(f"Epoch {epoch+1:3d} | train_loss: {train_loss:.4f} | "
                  f"val_loss: {val_loss:.4f} | val_auroc: {val_auroc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save checkpoint
                ckpt_path = self.output_dir / f"model_{self.model_type}.pt"
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  → Saved to {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(" Training complete")
    
    def evaluate(self):
        """Evaluate on test set."""
        print("\nEvaluating on test set...")
        
        test_loss, test_preds, test_labels = self.validate(self.test_loader)
        
        metrics = compute_metrics(test_labels, test_preds)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Metrics:")
        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}")
        
        # Bootstrap
        print("\nBootstrap analysis...")
        boot_metrics = bootstrap_metrics(test_labels, test_preds, n_bootstrap=100)
        
        # Save results
        results = {
            'loss': float(test_loss),
            'metrics': metrics,
            'bootstrap_metrics': {
                k: {'mean': float(v['mean']), 'std': float(v['std']), 'ci95': v['ci95']}
                for k, v in boot_metrics.items()
            }
        }
        
        results_path = self.output_dir / f"results_{self.model_type}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save predictions
        pred_df = pd.DataFrame({
            'predictions': test_preds,
            'labels': test_labels
        })
        pred_df.to_csv(self.output_dir / f"predictions_{self.model_type}.csv", index=False)
        
        # Generate report
        report = MetricsReporter.generate_summary(test_preds, test_labels)
        report_path = self.output_dir / f"summary_{self.model_type}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f" Results saved to {self.output_dir}")
    
    def run(self):
        """Run full pipeline."""
        try:
            self.load_data()
            self.create_dataloaders()
            self.build_model()
            self.train()
            self.evaluate()
            
            print(f"\n{'='*60}")
            print("✅ Pipeline complete!")
            print(f"Results saved to: {self.output_dir}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train quantum drug-protein interaction model")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file')
    parser.add_argument('--model-type', type=str, default='full', choices=['full', 'classical'])
    
    args = parser.parse_args()
    
    pipeline = Pipeline(args.config, model_type=args.model_type)
    pipeline.run()


if __name__ == '__main__':
    main()
