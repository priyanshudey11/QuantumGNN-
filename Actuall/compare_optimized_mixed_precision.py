"""
Optimized QGNN Training with Mixed Precision
==============================================
This script implements automatic mixed precision (AMP) training which can provide
2-3x speedup on CUDA GPUs without accuracy loss.

Key optimizations:
1. Mixed Precision (FP16) for faster GPU computation
2. Gradient scaling to prevent underflow
3. Optimized data types throughout
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import multiprocessing as mp

# Mixed precision imports
from torch.cuda.amp import autocast, GradScaler

from ligand_pocket_qgnn.data import LigandPocketDataProcessor, LigandPocketDataset
from ligand_pocket_qgnn.model import LigandPocketQGNN


# ========== OPTIMIZED COLLATE FUNCTION ==========
def optimized_collate_fn(batch):
    """
    Faster collate function using vectorized PyTorch operations.
    """
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


# ========== MIXED PRECISION TRAINING FUNCTIONS ==========
def train_epoch_amp(model, optimizer, criterion, loader, device, scaler, show_progress=True):
    """
    Train one epoch with Automatic Mixed Precision (AMP).
    Uses FP16 for faster computation while maintaining accuracy.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc='Training (AMP)', disable=not show_progress)

    for batch_idx, (x_batch, edge_index_batch, batch_vec, pocket_batch, labels) in enumerate(pbar):
        batch_start = datetime.now()

        # Non-blocking transfer to GPU
        x_batch = x_batch.to(device, non_blocking=True)
        edge_index_batch = edge_index_batch.to(device, non_blocking=True)
        batch_vec = batch_vec.to(device, non_blocking=True)
        pocket_batch = pocket_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ⭐ MIXED PRECISION FORWARD PASS
        with autocast():
            outputs = model(x_batch, edge_index_batch, batch_vec, pocket_batch).squeeze()
            loss = criterion(outputs, labels)

        # ⭐ SCALED BACKWARD PASS
        scaler.scale(loss).backward()

        # ⭐ UNSCALE GRADIENTS BEFORE CLIPPING
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ⭐ SCALED OPTIMIZER STEP
        scaler.step(optimizer)
        scaler.update()

        # Metrics (convert to float for accumulation)
        total_loss += loss.float().item() * len(labels)
        all_preds.extend(outputs.float().detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        batch_time = (datetime.now() - batch_start).total_seconds()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'time': f'{batch_time:.1f}s',
            'scale': f'{scaler.get_scale():.0f}'
        })

    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))

    return {'loss': avg_loss, 'accuracy': accuracy}


def evaluate_amp(model, criterion, loader, device):
    """
    Evaluate model with mixed precision.
    """
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

            # ⭐ MIXED PRECISION INFERENCE
            with autocast():
                outputs = model(x_batch, edge_index_batch, batch_vec, pocket_batch).squeeze()
                loss = criterion(outputs, labels)

            total_loss += loss.float().item() * len(labels)
            all_preds.extend(outputs.float().cpu().numpy())
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


def train_model_amp(model, train_loader, val_loader, learning_rate, model_name, device,
                    save_dir, epochs, patience, resume_from_checkpoint=None):
    """
    Train with Automatic Mixed Precision (AMP) and detailed progress tracking.
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name.upper()} Model with MIXED PRECISION")
    print(f"{'='*70}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"⭐ Mixed Precision: ENABLED (FP16)")
    print(f"{'='*70}\n")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = nn.BCELoss()

    # ⭐ CREATE GRADIENT SCALER FOR MIXED PRECISION
    scaler = GradScaler()

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'learning_rates': []
    }

    best_val_auc = 0.0
    patience_counter = 0
    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_from_checkpoint is not None:
        checkpoint_path = os.path.join(save_dir, f"{model_name}_best_amp.pt")
        history_path = os.path.join(save_dir, f"{model_name}_history_amp.json")

        if os.path.exists(checkpoint_path) and os.path.exists(history_path):
            print(f"{'='*70}")
            print(f"RESUMING FROM CHECKPOINT")
            print(f"{'='*70}")

            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            best_val_auc = checkpoint['best_auc']

            with open(history_path, 'r') as f:
                history = json.load(f)

            start_epoch = len(history['train_loss'])
            best_epoch = np.argmax(history['val_auc'])
            patience_counter = start_epoch - 1 - best_epoch

            print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
            print(f"✓ Best AUC so far: {best_val_auc:.4f} (epoch {best_epoch + 1})")
            print(f"✓ Resuming from epoch {start_epoch + 1}")
            print(f"✓ Current patience: {patience_counter}/{patience}")
            print(f"✓ GradScaler state restored")

            for auc in history['val_auc']:
                scheduler.step(auc)

            print(f"✓ Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'='*70}\n")

    start_time = datetime.now()

    for epoch in range(start_epoch, epochs):
        epoch_start = datetime.now()
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{epochs} - Started at {epoch_start.strftime('%H:%M:%S')}")
        print(f"{'='*70}")

        # ⭐ USE AMP TRAINING FUNCTION
        train_metrics = train_epoch_amp(model, optimizer, criterion, train_loader, device, scaler)
        val_metrics = evaluate_amp(model, criterion, val_loader, device)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['auc'])
        current_lr = optimizer.param_groups[0]['lr']

        if current_lr != old_lr:
            print(f"  Learning rate reduced: {old_lr:.6f} → {current_lr:.6f}")

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

        print(f"\n{'-'*70}")
        print(f"Epoch {epoch+1} Results ({epoch_time:.1f}s):")
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}")
        print(f"  Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}, "
              f"auc={val_metrics['auc']:.4f}, f1={val_metrics['f1']:.4f}")
        print(f"  Best AUC so far: {best_val_auc:.4f}")
        print(f"  GradScaler: {scaler.get_scale():.0f}")
        print(f"{'-'*70}")

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0

            # ⭐ SAVE SCALER STATE
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_auc': best_val_auc,
                'history': history
            }, os.path.join(save_dir, f"{model_name}_best_amp.pt"))

            print(f"✓ New best AUC: {best_val_auc:.4f} (saved)")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")

        with open(os.path.join(save_dir, f"{model_name}_history_amp.json"), 'w') as f:
            json.dump(history, f, indent=2)

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{model_name.upper()} Training Complete!")
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Best AUC: {best_val_auc:.4f}")
    print(f"  Average epoch time: {total_time/(epoch-start_epoch+1):.1f}s")

    return history, best_val_auc


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "/media/priyanshu/SD/othercode/data"
    SAVE_DIR = "./ligand_pocket_comparison_results"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Parameters
    MAX_SAMPLES = 0
    SEED = 42069
    HIDDEN_DIM = 64
    N_QUBITS = 8
    N_QLAYERS = 4
    QUANTUM_DEVICE = 'lightning.gpu'
    BATCH_SIZE = 8192
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 15

    # Setup
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CPU_COUNT = mp.cpu_count()
    NUM_WORKERS = max(4, CPU_COUNT - 2)
    PREFETCH_FACTOR = 4

    print("="*70)
    print("OPTIMIZED QGNN TRAINING WITH MIXED PRECISION")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Mixed Precision: {'ENABLED (FP16)' if DEVICE == 'cuda' else 'DISABLED (CPU only)'}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Qubits: {N_QUBITS}, Layers: {N_QLAYERS}")
    print("="*70)

    # Load data
    print("\nLoading data...")
    processor = LigandPocketDataProcessor(DATA_DIR, seed=SEED)
    processor.load_data(max_samples=MAX_SAMPLES)
    interactions = processor.get_dataset()

    # Determine dimensions
    sample_ligand = processor.ligands[interactions[0].ligand_id]
    sample_pocket = processor.pockets[interactions[0].pocket_id]
    ligand_dim = sample_ligand.atom_features.shape[1]
    pocket_dim = sample_pocket.to_vector().shape[0]

    # Create datasets
    train_ints, val_ints = train_test_split(interactions, test_size=VAL_SPLIT, random_state=SEED)
    train_dataset = LigandPocketDataset(processor, train_ints)
    val_dataset = LigandPocketDataset(processor, val_ints)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=optimized_collate_fn, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True, prefetch_factor=PREFETCH_FACTOR
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=optimized_collate_fn, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True, prefetch_factor=PREFETCH_FACTOR
    )

    # Train Quantum Model with AMP
    print("\nCreating Quantum Model...")
    quantum_model = LigandPocketQGNN(
        ligand_in_dim=ligand_dim,
        pocket_in_dim=pocket_dim,
        hidden_dim=HIDDEN_DIM,
        n_qubits=N_QUBITS,
        n_qlayers=N_QLAYERS,
        use_quantum=True,
        quantum_device=QUANTUM_DEVICE
    )

    quantum_history, quantum_best_auc = train_model_amp(
        quantum_model, train_loader, val_loader,
        LEARNING_RATE, "quantum", DEVICE, SAVE_DIR,
        EPOCHS, EARLY_STOPPING_PATIENCE,
        resume_from_checkpoint=True
    )

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Quantum AUC: {quantum_best_auc:.4f}")
    print(f"Results saved to: {SAVE_DIR}")
    print(f"{'='*70}")
