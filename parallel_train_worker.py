"""
Worker script for parallel hyperparameter training.
This script trains a single configuration and is called by the main notebook.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
from torch.utils.data import DataLoader
import time

from data import LigandPocketDataProcessor, LigandPocketDataset, collate_fn
from model import LigandPocketQGNN


def train_single_config(config_num, hidden_dim, n_qubits, n_qlayers, learning_rate,
                       data_dir, save_dir, seed=42069, epochs=100,
                       early_stopping_patience=15, batch_size=512):
    """
    Train a single hyperparameter configuration.
    Returns dict with results.
    """
    # Set different random seed for each process
    process_seed = seed + config_num
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)

    # Use CPU for parallel training to avoid MPS conflicts
    device = torch.device('cpu')

    start_time = time.time()

    try:
        # Create unique model name
        model_name = f"quantum_hd{hidden_dim}_q{n_qubits}_ql{n_qlayers}_lr{learning_rate}"

        print(f"[Config {config_num}] Starting: {model_name}")

        # Load data
        proc = LigandPocketDataProcessor(data_dir, seed=seed)
        proc.load_data(max_samples=50)
        ints = proc.get_dataset()

        tr_ints, tmp_ints = train_test_split(ints, test_size=0.2, random_state=seed)
        v_ints, ts_ints = train_test_split(tmp_ints, test_size=0.5, random_state=seed)

        train_dataset = LigandPocketDataset(proc, tr_ints)
        val_dataset = LigandPocketDataset(proc, v_ints)
        test_dataset = LigandPocketDataset(proc, ts_ints)

        # Get dimensions
        sample_ligand = proc.ligands[ints[0].ligand_id]
        sample_pocket = proc.pockets[ints[0].pocket_id]
        ligand_dim = sample_ligand.atom_features.shape[1]
        pocket_dim = sample_pocket.to_vector().shape[0]

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=0)

        # Create model
        model = LigandPocketQGNN(
            ligand_in_dim=ligand_dim,
            pocket_in_dim=pocket_dim,
            hidden_dim=hidden_dim,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
            use_quantum=True,
            quantum_device='lightning.qubit'
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                               factor=0.5, patience=5)
        criterion = nn.BCELoss()

        best_val_auc = 0.0
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            # Train
            model.train()
            for x, edge_idx, batch_vec, pocket, labels in train_loader:
                x, edge_idx, batch_vec, pocket, labels = (
                    x.to(device), edge_idx.to(device), batch_vec.to(device),
                    pocket.to(device), labels.to(device)
                )

                optimizer.zero_grad(set_to_none=True)
                outputs = model(x, edge_idx, batch_vec, pocket).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validate
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, edge_idx, batch_vec, pocket, labels in val_loader:
                    x, edge_idx, batch_vec, pocket, labels = (
                        x.to(device), edge_idx.to(device), batch_vec.to(device),
                        pocket.to(device), labels.to(device)
                    )
                    outputs = model(x, edge_idx, batch_vec, pocket).squeeze()
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_auc = roc_auc_score(all_labels, all_preds)
            scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best.pt"))
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                break

        # Test on best model
        model.load_state_dict(torch.load(os.path.join(save_dir, f"{model_name}_best.pt")))
        model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for x, edge_idx, batch_vec, pocket, labels in test_loader:
                x, edge_idx, batch_vec, pocket, labels = (
                    x.to(device), edge_idx.to(device), batch_vec.to(device),
                    pocket.to(device), labels.to(device)
                )
                outputs = model(x, edge_idx, batch_vec, pocket).squeeze()
                loss = criterion(outputs, labels)
                total_loss += loss.item() * len(labels)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_preds_binary = (all_preds >= 0.5).astype(int)

        # Calculate metrics
        test_metrics = {
            'loss': total_loss / len(all_labels),
            'accuracy': accuracy_score(all_labels, all_preds_binary),
            'auc': roc_auc_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds_binary, zero_division=0),
            'recall': recall_score(all_labels, all_preds_binary, zero_division=0),
            'f1': f1_score(all_labels, all_preds_binary, zero_division=0)
        }

        training_time = time.time() - start_time

        result = {
            'config_num': config_num,
            'hidden_dim': hidden_dim,
            'n_qubits': n_qubits,
            'n_qlayers': n_qlayers,
            'learning_rate': learning_rate,
            'val_auc': best_val_auc,
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_auc': test_metrics['auc'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'model_params': sum(p.numel() for p in model.parameters()),
            'training_time_sec': training_time,
            'status': 'success'
        }

        print(f"[Config {config_num}] ✓ DONE: {model_name} - Test AUC: {test_metrics['auc']:.4f} ({training_time/60:.1f} min)")
        return result

    except Exception as e:
        print(f"[Config {config_num}] ✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'config_num': config_num,
            'hidden_dim': hidden_dim,
            'n_qubits': n_qubits,
            'n_qlayers': n_qlayers,
            'learning_rate': learning_rate,
            'status': 'failed',
            'error': str(e)
        }


if __name__ == '__main__':
    # Allow running this script directly for testing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_num', type=int, required=True)
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--n_qubits', type=int, required=True)
    parser.add_argument('--n_qlayers', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()

    result = train_single_config(
        args.config_num, args.hidden_dim, args.n_qubits, args.n_qlayers,
        args.learning_rate, args.data_dir, args.save_dir
    )

    print(f"\nFinal result: {result}")
