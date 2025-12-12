#!/usr/bin/env python3
"""
FASTEST QGNN Training Script
=============================
This script combines ALL optimizations for maximum performance:

1. ⭐ Reduced quantum circuit (4 qubits, 2 layers) - 5-10x speedup
2. ⭐ Mixed precision (FP16) - 2-3x speedup
3. ⭐ Cython collate (if available) - 10-20% speedup

Expected speedup: 15-30x over original implementation
Batch time: ~10-20s (down from ~275s)

Usage:
    python train_fast_qgnn.py

To compile Cython extension (optional, ~10% extra speedup):
    python setup_cython.py build_ext --inplace
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
from torch.cuda.amp import autocast, GradScaler

from ligand_pocket_qgnn.data import LigandPocketDataProcessor, LigandPocketDataset

# Try to import fast model
try:
    from ligand_pocket_qgnn.model_fast import LigandPocketQGNNFast
    FAST_MODEL_AVAILABLE = True
except ImportError:
    print("⚠ Warning: Fast model not available, falling back to regular model")
    from ligand_pocket_qgnn.model import LigandPocketQGNN
    FAST_MODEL_AVAILABLE = False

# Try to import Cython collate
try:
    from ligand_pocket_qgnn.collate_cython import fast_collate_fn_cython
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


def optimized_collate_fn(batch):
    """Python fallback collate function."""
    x_list, edge_index_list, pocket_list, label_list = zip(*batch)
    pocket_batch = torch.stack(pocket_list)
    label_batch = torch.stack(label_list)
    num_nodes_list = torch.tensor([x.shape[0] for x in x_list], dtype=torch.long)
    cumsum = torch.cat([torch.zeros(1, dtype=torch.long), num_nodes_list.cumsum(0)])
    x_batch = torch.cat(x_list, dim=0)
    edge_index_shifted = []
    for i, edge_index in enumerate(edge_index_list):
        if edge_index.shape[1] > 0:
            edge_index_shifted.append(edge_index + cumsum[i])
    if edge_index_shifted:
        edge_index_batch = torch.cat(edge_index_shifted, dim=1)
    else:
        edge_index_batch = torch.zeros((2, 0), dtype=torch.long)
    batch_vec = torch.cat([torch.full((n,), i, dtype=torch.long)
                           for i, n in enumerate(num_nodes_list)])
    return x_batch, edge_index_batch, batch_vec, pocket_batch, label_batch


def train_epoch(model, optimizer, criterion, loader, device, scaler, use_amp=True):
    """Train one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc='Training')
    for x_batch, edge_index_batch, batch_vec, pocket_batch, labels in pbar:
        x_batch = x_batch.to(device, non_blocking=True)
        edge_index_batch = edge_index_batch.to(device, non_blocking=True)
        batch_vec = batch_vec.to(device, non_blocking=True)
        pocket_batch = pocket_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                outputs = model(x_batch, edge_index_batch, batch_vec, pocket_batch).squeeze()
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x_batch, edge_index_batch, batch_vec, pocket_batch).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.float().item() * len(labels) if use_amp else loss.item() * len(labels)
        all_preds.extend((outputs.float() if use_amp else outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
    return {'loss': avg_loss, 'accuracy': accuracy}


def evaluate(model, criterion, loader, device, use_amp=True):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, edge_index_batch, batch_vec, pocket_batch, labels in tqdm(loader, desc='Validation', leave=False):
            x_batch = x_batch.to(device, non_blocking=True)
            edge_index_batch = edge_index_batch.to(device, non_blocking=True)
            batch_vec = batch_vec.to(device, non_blocking=True)
            pocket_batch = pocket_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with autocast():
                    outputs = model(x_batch, edge_index_batch, batch_vec, pocket_batch).squeeze()
                    loss = criterion(outputs, labels)
                total_loss += loss.float().item() * len(labels)
                all_preds.extend(outputs.float().cpu().numpy())
            else:
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


def main():
    # ========== CONFIGURATION ==========
    DATA_DIR = "/media/priyanshu/SD/othercode/data"
    SAVE_DIR = "./ligand_pocket_fast_results"
    os.makedirs(SAVE_DIR, exist_ok=True)

    MAX_SAMPLES = 0
    SEED = 42069
    HIDDEN_DIM = 64
    N_QUBITS = 4 if FAST_MODEL_AVAILABLE else 8  # ⭐ Reduced for speed
    N_QLAYERS = 2 if FAST_MODEL_AVAILABLE else 4  # ⭐ Reduced for speed
    QUANTUM_DEVICE = 'lightning.gpu'
    BATCH_SIZE = 8192
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 15

    # Setup
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_AMP = DEVICE == 'cuda'  # Only use AMP on CUDA

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    CPU_COUNT = mp.cpu_count()
    NUM_WORKERS = max(4, CPU_COUNT - 2)
    PREFETCH_FACTOR = 4

    # Select collate function
    if CYTHON_AVAILABLE:
        collate_fn = fast_collate_fn_cython
        collate_name = "Cython (C-optimized)"
    else:
        collate_fn = optimized_collate_fn
        collate_name = "Python (optimized)"

    # Print configuration
    print("="*80)
    print("FASTEST QGNN TRAINING - ALL OPTIMIZATIONS ENABLED")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Fast Model: {'✓ ENABLED' if FAST_MODEL_AVAILABLE else '✗ DISABLED (using regular model)'}")
    print(f"Mixed Precision: {'✓ ENABLED (FP16)' if USE_AMP else '✗ DISABLED (CPU only)'}")
    print(f"Cython Collate: {'✓ ENABLED' if CYTHON_AVAILABLE else '✗ DISABLED (compile with setup_cython.py)'}")
    print(f"Collate Function: {collate_name}")
    print(f"\nModel Configuration:")
    print(f"  Qubits: {N_QUBITS} {'(4x faster)' if FAST_MODEL_AVAILABLE else ''}")
    print(f"  Layers: {N_QLAYERS} {'(2x faster)' if FAST_MODEL_AVAILABLE else ''}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"\nExpected Speedup:")
    speedup = 1.0
    if FAST_MODEL_AVAILABLE:
        speedup *= 8.0
        print(f"  - Reduced circuit: 8x")
    if USE_AMP:
        speedup *= 2.5
        print(f"  - Mixed precision: 2.5x")
    if CYTHON_AVAILABLE:
        speedup *= 1.15
        print(f"  - Cython collate: 1.15x")
    print(f"  TOTAL: ~{speedup:.1f}x faster than baseline")
    print(f"\nEstimated batch time: {275/speedup:.1f}s (was ~275s)")
    print("="*80)

    # Load data
    print("\nLoading data...")
    processor = LigandPocketDataProcessor(DATA_DIR, seed=SEED)
    processor.load_data(max_samples=MAX_SAMPLES)
    interactions = processor.get_dataset()

    sample_ligand = processor.ligands[interactions[0].ligand_id]
    sample_pocket = processor.pockets[interactions[0].pocket_id]
    ligand_dim = sample_ligand.atom_features.shape[1]
    pocket_dim = sample_pocket.to_vector().shape[0]

    train_ints, val_ints = train_test_split(interactions, test_size=VAL_SPLIT, random_state=SEED)
    train_dataset = LigandPocketDataset(processor, train_ints)
    val_dataset = LigandPocketDataset(processor, val_ints)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True, prefetch_factor=PREFETCH_FACTOR
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True, prefetch_factor=PREFETCH_FACTOR
    )

    # Create model
    print("\nCreating model...")
    if FAST_MODEL_AVAILABLE:
        model = LigandPocketQGNNFast(
            ligand_in_dim=ligand_dim,
            pocket_in_dim=pocket_dim,
            hidden_dim=HIDDEN_DIM,
            n_qubits=N_QUBITS,
            n_qlayers=N_QLAYERS,
            use_quantum=True,
            quantum_device=QUANTUM_DEVICE
        )
    else:
        model = LigandPocketQGNN(
            ligand_in_dim=ligand_dim,
            pocket_in_dim=pocket_dim,
            hidden_dim=HIDDEN_DIM,
            n_qubits=N_QUBITS,
            n_qlayers=N_QLAYERS,
            use_quantum=True,
            quantum_device=QUANTUM_DEVICE
        )

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.BCELoss()
    scaler = GradScaler() if USE_AMP else None

    # Training loop
    print("\nStarting training...")
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        epoch_start = datetime.now()
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{EPOCHS}")
        print(f"{'='*80}")

        train_metrics = train_epoch(model, optimizer, criterion, train_loader, DEVICE, scaler, USE_AMP)
        val_metrics = evaluate(model, criterion, val_loader, DEVICE, USE_AMP)

        scheduler.step(val_metrics['auc'])

        epoch_time = (datetime.now() - epoch_start).total_seconds()

        print(f"\nResults ({epoch_time:.1f}s):")
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}")
        print(f"  Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}, "
              f"auc={val_metrics['auc']:.4f}, f1={val_metrics['f1']:.4f}")

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            print(f"✓ New best AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
