#!/usr/bin/env python3
"""
Quantum vs Classical GNN - A6000 Optimized Training Script
==========================================================
Hardware: 2× NVIDIA A6000 (48GB VRAM each)
Expected runtime: 2-3 hours
"""

import os
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
import json
import sys

# ============================================================================
# CONFIGURATION - UPDATE THIS FOR YOUR SERVER!
# ============================================================================

DATA_DIR = "/media/priyanshu/SD/othercode/data"  # ⚠️ UPDATE THIS PATH!
SAVE_DIR = "./a6000_results"
LOG_FILE = "a6000_training.log"

# Data parameters
MAX_DRUGS = 2000
SEED = 42

# Model parameters
NUM_QUBITS = 6
NUM_QLAYERS = 3
HIDDEN_DIM = 256

# Training parameters
EPOCHS = 100
BATCH_SIZE = 2048
LEARNING_RATE_QUANTUM = 0.01
LEARNING_RATE_CLASSICAL = 0.001
VAL_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 15

# Hardware
DEVICE = 'cuda'
USE_MULTI_GPU = True
QUANTUM_DEVICE = 'lightning.qubit'
NUM_WORKERS = 16
PIN_MEMORY = True

# ============================================================================
# LOGGING SETUP
# ============================================================================

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(LOG_FILE)
sys.stderr = sys.stdout

print(f"{'='*80}")
print(f"🚀 A6000 Quantum vs Classical GNN Training")
print(f"{'='*80}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Expected runtime: 2-3 hours\n")

# ============================================================================
# IMPORT MODULES
# ============================================================================

from drug_patient_qgnn import (
    DrugPatientDataProcessor,
    QuantumDrugPatientGNN,
    set_seed,
    print_model_summary,
    print_device_info
)

set_seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

print_device_info()
print(f"\nGPU Count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1e9:.0f} GB)")

# ============================================================================
# DATA LOADING
# ============================================================================

print(f"\n{'='*80}")
print(f"LOADING DATA")
print(f"{'='*80}")
print(f"Data directory: {DATA_DIR}")
print(f"Max drugs: {MAX_DRUGS}\n")

start_time = datetime.now()

processor = DrugPatientDataProcessor(data_dir=DATA_DIR, seed=SEED)
counts = processor.load_real_data(data_dir=DATA_DIR, max_samples=MAX_DRUGS)

stats = processor.get_statistics()
print(f"\nData loaded in {(datetime.now() - start_time).total_seconds():.1f}s")
print("\nDataset Statistics:")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key:25s}: {value:.4f}")
    else:
        print(f"  {key:25s}: {value}")

# ============================================================================
# DATASET PREPARATION
# ============================================================================

graph = processor.graph
drug_features = graph.get_drug_features_matrix()
patient_features = graph.get_patient_features_matrix()
edge_index, edge_features = graph.get_edge_index()
labels = graph.get_edge_labels()

interaction_data = []
for idx in range(edge_index.shape[1]):
    drug_idx = int(edge_index[0, idx])
    patient_idx = int(edge_index[1, idx])

    interaction_data.append({
        'drug_features': drug_features[drug_idx].tolist(),
        'patient_features': patient_features[patient_idx].tolist(),
        'label': float(labels[idx])
    })

df_pandas = pd.DataFrame(interaction_data)

train_pd, val_pd = train_test_split(
    df_pandas,
    test_size=VAL_SPLIT,
    random_state=SEED,
    stratify=df_pandas['label']
)

batches_per_epoch = len(train_pd) // BATCH_SIZE
print(f"\nTraining samples:   {len(train_pd):,}")
print(f"Validation samples: {len(val_pd):,}")
print(f"Batches per epoch:  {batches_per_epoch}")

class InteractionDataset(Dataset):
    def __init__(self, df):
        self.drug_features = np.stack(df['drug_features'].values)
        self.patient_features = np.stack(df['patient_features'].values)
        self.labels = df['label'].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.drug_features[idx], dtype=torch.float32),
            torch.tensor(self.patient_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

train_dataset = InteractionDataset(train_pd)
val_dataset = InteractionDataset(val_pd)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=True
)

print(f"✓ DataLoaders ready with {NUM_WORKERS} workers")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, optimizer, criterion, loader, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for drug_features, patient_features, labels in loader:
        drug_features = drug_features.to(device, non_blocking=True)
        patient_features = patient_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(drug_features, patient_features).squeeze(-1)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))

    return {'loss': avg_loss, 'accuracy': accuracy}

def evaluate(model, criterion, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for drug_features, patient_features, labels in loader:
            drug_features = drug_features.to(device, non_blocking=True)
            patient_features = patient_features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(drug_features, patient_features).squeeze(-1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
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

def train_model(model, train_loader, val_loader, learning_rate, model_name, device, use_multi_gpu=False):
    print(f"\n{'='*80}")
    print(f"TRAINING {model_name.upper()} MODEL")
    print(f"{'='*80}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Device: {device}")
    print(f"Multi-GPU: {use_multi_gpu}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Wrap model in DataParallel for multi-GPU
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'learning_rates': []
    }

    best_val_auc = 0.0
    patience_counter = 0
    start_time = datetime.now()

    for epoch in range(EPOCHS):
        epoch_start = datetime.now()

        train_metrics = train_epoch(model, optimizer, criterion, train_loader, device)
        val_metrics = evaluate(model, criterion, val_loader, device)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['auc'])
        current_lr = optimizer.param_groups[0]['lr']

        if current_lr != old_lr:
            print(f"  LR reduced: {old_lr:.6f} → {current_lr:.6f}")

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
        elapsed_time = (datetime.now() - start_time).total_seconds()

        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Epoch {epoch+1:3d}/{EPOCHS} ({epoch_time/60:4.1f}min) - "
              f"loss: {train_metrics['loss']:.4f} - "
              f"val_auc: {val_metrics['auc']:.4f} "
              f"val_acc: {val_metrics['accuracy']:.4f} "
              f"[Best: {best_val_auc:.4f}] "
              f"[Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}] "
              f"[Elapsed: {elapsed_time/3600:.1f}h]")

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0

            # Save model (unwrap DataParallel if needed)
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_val_auc,
                'history': history
            }, os.path.join(SAVE_DIR, f"{model_name}_best.pt"))

            print(f"  ✓ New best AUC: {best_val_auc:.4f} (saved)")
        else:
            patience_counter += 1

        # Save history every 5 epochs
        if (epoch + 1) % 5 == 0:
            with open(os.path.join(SAVE_DIR, f"{model_name}_history.json"), 'w') as f:
                json.dump(history, f, indent=2)

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            break

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{model_name.upper()} Training Complete!")
    print(f"  Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"  Best AUC: {best_val_auc:.4f}")
    print(f"  Final epoch: {epoch+1}/{EPOCHS}")

    # Save final history
    with open(os.path.join(SAVE_DIR, f"{model_name}_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    return history, best_val_auc

# ============================================================================
# TRAIN QUANTUM MODEL
# ============================================================================

drug_dim = len(drug_features[0])
patient_dim = len(patient_features[0])

quantum_model = QuantumDrugPatientGNN(
    drug_dim=drug_dim,
    patient_dim=patient_dim,
    num_qubits=NUM_QUBITS,
    num_qlayers=NUM_QLAYERS,
    hidden_dim=HIDDEN_DIM,
    use_quantum=True,
    device_name=QUANTUM_DEVICE
)

print_model_summary(quantum_model, drug_dim, patient_dim)

quantum_history, quantum_best_auc = train_model(
    quantum_model,
    train_loader,
    val_loader,
    LEARNING_RATE_QUANTUM,
    "quantum",
    DEVICE,
    use_multi_gpu=False
)

# ============================================================================
# TRAIN CLASSICAL MODEL
# ============================================================================

classical_model = QuantumDrugPatientGNN(
    drug_dim=drug_dim,
    patient_dim=patient_dim,
    num_qubits=NUM_QUBITS,
    num_qlayers=NUM_QLAYERS,
    hidden_dim=HIDDEN_DIM,
    use_quantum=False
)

print_model_summary(classical_model, drug_dim, patient_dim)

classical_history, classical_best_auc = train_model(
    classical_model,
    train_loader,
    val_loader,
    LEARNING_RATE_CLASSICAL,
    "classical",
    DEVICE,
    use_multi_gpu=USE_MULTI_GPU
)

# ============================================================================
# GENERATE RESULTS
# ============================================================================

print(f"\n{'='*80}")
print(f"GENERATING RESULTS")
print(f"{'='*80}\n")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(quantum_history['val_loss'], label='Quantum', linewidth=2.5)
axes[0, 0].plot(classical_history['val_loss'], label='Classical', linewidth=2.5)
axes[0, 0].set_title('Validation Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(quantum_history['val_auc'], label='Quantum', linewidth=2.5)
axes[0, 1].plot(classical_history['val_auc'], label='Classical', linewidth=2.5)
axes[0, 1].set_title('Validation AUC-ROC (Primary Metric)')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(quantum_history['val_acc'], label='Quantum', linewidth=2.5)
axes[1, 0].plot(classical_history['val_acc'], label='Classical', linewidth=2.5)
axes[1, 0].set_title('Validation Accuracy')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(quantum_history['val_f1'], label='Quantum', linewidth=2.5)
axes[1, 1].plot(classical_history['val_f1'], label='Classical', linewidth=2.5)
axes[1, 1].set_title('Validation F1 Score')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'comparison_plots.png'), dpi=300, bbox_inches='tight')
print(f"✓ Plots saved to {os.path.join(SAVE_DIR, 'comparison_plots.png')}")

# Print final results
print(f"\n{'='*80}")
print(f"FINAL RESULTS: QUANTUM vs CLASSICAL")
print(f"{'='*80}\n")

print(f"{'Metric':<25} {'Quantum':<15} {'Classical':<15} {'Difference':<15} {'Winner'}")
print(f"{'-'*80}")

metrics = [
    ('Best Validation AUC', quantum_best_auc, classical_best_auc),
    ('Final Val Accuracy', quantum_history['val_acc'][-1], classical_history['val_acc'][-1]),
    ('Final Val F1 Score', quantum_history['val_f1'][-1], classical_history['val_f1'][-1]),
    ('Epochs Trained', len(quantum_history['val_auc']), len(classical_history['val_auc']))
]

quantum_wins = 0
for metric_name, quantum_val, classical_val in metrics:
    diff = quantum_val - classical_val
    if metric_name != 'Epochs Trained':
        diff_pct = (diff / classical_val) * 100
        winner = 'QUANTUM' if quantum_val > classical_val else 'Classical'
        if quantum_val > classical_val:
            quantum_wins += 1
        print(f"{metric_name:<25} {quantum_val:<15.4f} {classical_val:<15.4f} {diff:+.4f} ({diff_pct:+.1f}%)  {winner}")
    else:
        print(f"{metric_name:<25} {quantum_val:<15.0f} {classical_val:<15.0f} {diff:+.0f}")

print(f"\n{'='*80}")
if quantum_wins > 1:
    print(f"🏆 QUANTUM WINS: {quantum_wins}/3 metrics")
elif quantum_wins == 1:
    print(f"⚖️  TIE: Both models competitive")
else:
    print(f"🏆 CLASSICAL WINS: {3-quantum_wins}/3 metrics")
print(f"{'='*80}\n")

# Save results to file
results_text = f"""
QUANTUM VS CLASSICAL GNN - A6000 EXPERIMENT RESULTS
{'='*80}

Configuration:
- Hardware: 2× NVIDIA A6000 (48GB VRAM)
- Batch Size: {BATCH_SIZE}
- Dataset: {MAX_DRUGS} drugs
- Quantum Layers: {NUM_QLAYERS}
- Hidden Dim: {HIDDEN_DIM}

Results:
- Quantum Best AUC: {quantum_best_auc:.4f} ({len(quantum_history['val_auc'])} epochs)
- Classical Best AUC: {classical_best_auc:.4f} ({len(classical_history['val_auc'])} epochs)
- Difference: {quantum_best_auc - classical_best_auc:+.4f} ({((quantum_best_auc - classical_best_auc) / classical_best_auc * 100):+.1f}%)

Winner: {'QUANTUM' if quantum_best_auc > classical_best_auc else 'CLASSICAL'}

Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(os.path.join(SAVE_DIR, 'final_results.txt'), 'w') as f:
    f.write(results_text)

# Save JSON results
results_json = {
    'quantum': {
        'best_auc': float(quantum_best_auc),
        'final_accuracy': float(quantum_history['val_acc'][-1]),
        'final_f1': float(quantum_history['val_f1'][-1]),
        'epochs_trained': len(quantum_history['val_auc']),
        'history': quantum_history
    },
    'classical': {
        'best_auc': float(classical_best_auc),
        'final_accuracy': float(classical_history['val_acc'][-1]),
        'final_f1': float(classical_history['val_f1'][-1]),
        'epochs_trained': len(classical_history['val_auc']),
        'history': classical_history
    },
    'config': {
        'hardware': '2× A6000 (48GB)',
        'batch_size': BATCH_SIZE,
        'max_drugs': MAX_DRUGS,
        'num_qubits': NUM_QUBITS,
        'num_qlayers': NUM_QLAYERS,
        'hidden_dim': HIDDEN_DIM
    },
    'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open(os.path.join(SAVE_DIR, 'final_results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"✓ Results saved to {SAVE_DIR}/")
print(f"\n{'='*80}")
print(f"🎉 EXPERIMENT COMPLETE!")
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}\n")
