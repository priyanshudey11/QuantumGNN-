"""
Train Quantum and Classical models in PARALLEL to maximize hardware utilization.

This script trains both models simultaneously using multiprocessing.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import multiprocessing as mp

from ligand_pocket_qgnn.data import LigandPocketDataProcessor, LigandPocketDataset, collate_fn
from ligand_pocket_qgnn.model import LigandPocketQGNN

# Configuration
DATA_DIR = "/media/priyanshu/SD/othercode/data"
SAVE_DIR = "./ligand_pocket_comparison_results"
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42069
MAX_SAMPLES = 0
HIDDEN_DIM = 64
N_QUBITS = 6
N_QLAYERS = 2
BATCH_SIZE = 512  # Optimized for RTX 3080
NUM_WORKERS = 4   # Per model (total 8 workers)
EPOCHS = 100
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 15

def train_epoch(model, optimizer, criterion, loader, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for x_batch, edge_index_batch, batch_vec, pocket_batch, labels in tqdm(loader, desc='Training', leave=False):
        x_batch = x_batch.to(device)
        edge_index_batch = edge_index_batch.to(device)
        batch_vec = batch_vec.to(device)
        pocket_batch = pocket_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch, edge_index_batch, batch_vec, pocket_batch).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
    return {'loss': avg_loss, 'accuracy': accuracy}

def evaluate(model, criterion, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, edge_index_batch, batch_vec, pocket_batch, labels in tqdm(loader, desc='Validation', leave=False):
            x_batch = x_batch.to(device)
            edge_index_batch = edge_index_batch.to(device)
            batch_vec = batch_vec.to(device)
            pocket_batch = pocket_batch.to(device)
            labels = labels.to(device)

            outputs = model(x_batch, edge_index_batch, batch_vec, pocket_batch).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_preds_binary = (all_preds >= 0.5).astype(int)

    return {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds_binary),
        'auc': roc_auc_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds_binary, zero_division=0),
        'recall': recall_score(all_labels, all_preds_binary, zero_division=0),
        'f1': f1_score(all_labels, all_preds_binary, zero_division=0)
    }

def train_model_worker(model_type, ligand_dim, pocket_dim, train_ints, val_ints, processor_data):
    """Worker function to train a single model."""

    # Set device to GPU
    device = 'cuda'

    # Recreate processor (can't pickle it)
    processor = LigandPocketDataProcessor(DATA_DIR, seed=SEED)
    processor.pockets = processor_data['pockets']
    processor.ligands = processor_data['ligands']

    # Create datasets
    train_dataset = LigandPocketDataset(processor, train_ints)
    val_dataset = LigandPocketDataset(processor, val_ints)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )

    # Create model
    use_quantum = (model_type == 'quantum')
    model = LigandPocketQGNN(
        ligand_in_dim=ligand_dim,
        pocket_in_dim=pocket_dim,
        hidden_dim=HIDDEN_DIM,
        n_qubits=N_QUBITS,
        n_qlayers=N_QLAYERS,
        use_quantum=use_quantum
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.BCELoss()

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'learning_rates': []
    }

    best_val_auc = 0.0
    patience_counter = 0
    start_time = datetime.now()

    print(f"\n[{model_type.upper()}] Starting training at {start_time.strftime('%H:%M:%S')}")
    print(f"[{model_type.upper()}] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(EPOCHS):
        epoch_start = datetime.now()

        train_metrics = train_epoch(model, optimizer, criterion, train_loader, device)
        val_metrics = evaluate(model, criterion, val_loader, device)

        scheduler.step(val_metrics['auc'])
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['learning_rates'].append(current_lr)

        epoch_time = (datetime.now() - epoch_start).total_seconds()

        print(f"[{model_type.upper()}] Epoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s): "
              f"train_loss={train_metrics['loss']:.4f}, val_auc={val_metrics['auc']:.4f}")

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_val_auc,
                'history': history
            }, os.path.join(SAVE_DIR, f"{model_type}_best.pt"))
            print(f"[{model_type.upper()}] ✓ New best AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1

        with open(os.path.join(SAVE_DIR, f"{model_type}_history.json"), 'w') as f:
            json.dump(history, f, indent=2)

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"[{model_type.upper()}] Early stopping at epoch {epoch+1}")
            break

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n[{model_type.upper()}] Training Complete!")
    print(f"[{model_type.upper()}] Total time: {total_time/60:.2f} minutes")
    print(f"[{model_type.upper()}] Best AUC: {best_val_auc:.4f}")

    return model_type, history, best_val_auc

def main():
    """Main function to train both models in parallel."""

    print("="*70)
    print("PARALLEL TRAINING: Quantum + Classical Models")
    print("="*70)

    # Load data
    print("\nLoading data...")
    processor = LigandPocketDataProcessor(DATA_DIR, seed=SEED)
    processor.load_data(max_samples=MAX_SAMPLES)
    interactions = processor.get_dataset()

    # Split data
    train_ints, val_ints = train_test_split(interactions, test_size=VAL_SPLIT, random_state=SEED)

    # Get dimensions
    sample_ligand = processor.ligands[interactions[0].ligand_id]
    sample_pocket = processor.pockets[interactions[0].pocket_id]
    ligand_dim = sample_ligand.atom_features.shape[1]
    pocket_dim = sample_pocket.to_vector().shape[0]

    # Prepare processor data for pickling
    processor_data = {
        'pockets': processor.pockets,
        'ligands': processor.ligands
    }

    print(f"\nTotal samples: {len(interactions)}")
    print(f"Train: {len(train_ints)}, Val: {len(val_ints)}")
    print(f"Ligand dim: {ligand_dim}, Pocket dim: {pocket_dim}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers per model: {NUM_WORKERS}")

    # Train both models in parallel using multiprocessing
    print("\n" + "="*70)
    print("Starting PARALLEL training...")
    print("="*70)

    with mp.Pool(processes=2) as pool:
        results = pool.starmap(train_model_worker, [
            ('quantum', ligand_dim, pocket_dim, train_ints, val_ints, processor_data),
            ('classical', ligand_dim, pocket_dim, train_ints, val_ints, processor_data)
        ])

    # Process results
    quantum_result = [r for r in results if r[0] == 'quantum'][0]
    classical_result = [r for r in results if r[0] == 'classical'][0]

    _, quantum_history, quantum_best_auc = quantum_result
    _, classical_history, classical_best_auc = classical_result

    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"Quantum Model     - Best AUC: {quantum_best_auc:.4f}")
    print(f"Classical Model   - Best AUC: {classical_best_auc:.4f}")
    advantage = (quantum_best_auc - classical_best_auc) * 100
    print(f"\nQuantum Advantage: {advantage:.2f}% {'improvement' if advantage > 0 else 'deficit'}")
    print("="*70)

if __name__ == '__main__':
    # Set start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()
