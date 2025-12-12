#!/usr/bin/env python3
"""
Benchmark script to measure actual speedup from optimizations.

This script creates dummy data and benchmarks:
1. Original model (8 qubits, 4 layers, FP32)
2. Fast model (4 qubits, 2 layers, FP32)
3. Fast model with mixed precision (4 qubits, 2 layers, FP16)

Run: python benchmark_speedup.py
"""

import torch
import torch.nn as nn
import time
import numpy as np
from torch.cuda.amp import autocast, GradScaler

print("="*80)
print("QGNN SPEEDUP BENCHMARK")
print("="*80)

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

if device == 'cpu':
    print("\n⚠ WARNING: Running on CPU. Mixed precision won't help.")
    print("For best results, run on GPU.\n")

# Import models
try:
    from ligand_pocket_qgnn.model import LigandPocketQGNN
    from ligand_pocket_qgnn.model_fast import LigandPocketQGNNFast
    MODELS_AVAILABLE = True
except ImportError:
    print("\n❌ ERROR: Models not found. Make sure you're in the correct directory.")
    MODELS_AVAILABLE = False
    exit(1)

# Create dummy data
print("\nCreating dummy data...")
BATCH_SIZE = 1024
NUM_NODES = 50
NUM_EDGES = 100
LIGAND_DIM = 10
POCKET_DIM = 19

def create_dummy_batch():
    """Create a dummy batch for benchmarking."""
    x = torch.randn(NUM_NODES, LIGAND_DIM)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))
    batch = torch.zeros(NUM_NODES, dtype=torch.long)
    pocket = torch.randn(BATCH_SIZE, POCKET_DIM)
    labels = torch.randint(0, 2, (BATCH_SIZE,)).float()
    return x, edge_index, batch, pocket, labels

# Benchmark function
def benchmark_model(model, name, use_amp=False, num_iterations=10, warmup=3):
    """Benchmark a model."""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {name}")
    print(f"{'='*80}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    scaler = GradScaler() if use_amp else None

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        x, edge_index, batch, pocket, labels = create_dummy_batch()
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        pocket = pocket.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(x, edge_index, batch, pocket).squeeze()
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x, edge_index, batch, pocket).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Actual benchmark
    print(f"Running benchmark ({num_iterations} iterations)...")
    times = []

    for i in range(num_iterations):
        x, edge_index, batch, pocket, labels = create_dummy_batch()
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        pocket = pocket.to(device)
        labels = labels.to(device)

        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(x, edge_index, batch, pocket).squeeze()
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x, edge_index, batch, pocket).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = time.time() - start
        times.append(elapsed)

        print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.3f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nResults:")
    print(f"  Average time: {avg_time:.3f} ± {std_time:.3f}s")
    print(f"  Min time: {np.min(times):.3f}s")
    print(f"  Max time: {np.max(times):.3f}s")

    return avg_time


# Run benchmarks
print("\n" + "="*80)
print("STARTING BENCHMARKS")
print("="*80)

results = {}

# 1. Original model
print("\n[1/3] Original Model (8 qubits, 4 layers, FP32)")
original_model = LigandPocketQGNN(
    ligand_in_dim=LIGAND_DIM,
    pocket_in_dim=POCKET_DIM,
    hidden_dim=64,
    n_qubits=8,
    n_qlayers=4,
    use_quantum=True,
    quantum_device='lightning.gpu' if device == 'cuda' else 'lightning.qubit'
)
results['Original (FP32)'] = benchmark_model(original_model, "Original Model", use_amp=False, num_iterations=5, warmup=2)

# 2. Fast model
print("\n[2/3] Fast Model (4 qubits, 2 layers, FP32)")
fast_model = LigandPocketQGNNFast(
    ligand_in_dim=LIGAND_DIM,
    pocket_in_dim=POCKET_DIM,
    hidden_dim=64,
    n_qubits=4,
    n_qlayers=2,
    use_quantum=True,
    quantum_device='lightning.gpu' if device == 'cuda' else 'lightning.qubit'
)
results['Fast (FP32)'] = benchmark_model(fast_model, "Fast Model", use_amp=False, num_iterations=5, warmup=2)

# 3. Fast model with mixed precision (GPU only)
if device == 'cuda':
    print("\n[3/3] Fast Model with Mixed Precision (4 qubits, 2 layers, FP16)")
    fast_model_amp = LigandPocketQGNNFast(
        ligand_in_dim=LIGAND_DIM,
        pocket_in_dim=POCKET_DIM,
        hidden_dim=64,
        n_qubits=4,
        n_qlayers=2,
        use_quantum=True,
        quantum_device='lightning.gpu'
    )
    results['Fast + AMP (FP16)'] = benchmark_model(fast_model_amp, "Fast Model + AMP", use_amp=True, num_iterations=5, warmup=2)
else:
    print("\n[3/3] Skipping mixed precision (CPU only)")

# Print comparison
print("\n" + "="*80)
print("BENCHMARK RESULTS")
print("="*80)

baseline = results['Original (FP32)']

print(f"\n{'Configuration':<30} {'Time (s)':<15} {'Speedup':<15}")
print("-" * 80)

for name, time_taken in results.items():
    speedup = baseline / time_taken
    print(f"{name:<30} {time_taken:>10.3f}      {speedup:>10.2f}x")

print("-" * 80)

# Calculate combined speedup
if 'Fast + AMP (FP16)' in results:
    total_speedup = baseline / results['Fast + AMP (FP16)']
    print(f"\n🚀 TOTAL SPEEDUP: {total_speedup:.2f}x")
    print(f"   Batch time reduced from {baseline:.3f}s to {results['Fast + AMP (FP16)']:.3f}s")
else:
    total_speedup = baseline / results['Fast (FP32)']
    print(f"\n🚀 TOTAL SPEEDUP: {total_speedup:.2f}x (CPU only)")
    print(f"   Batch time reduced from {baseline:.3f}s to {results['Fast (FP32)']:.3f}s")

print("\n" + "="*80)
print("BENCHMARK COMPLETE")
print("="*80)

# Extrapolate to full training
print("\n📊 EXTRAPOLATION TO FULL TRAINING (24 batches/epoch, 25 epochs):")
print("-" * 80)

batches_per_epoch = 24
num_epochs = 25

for name, time_per_batch in results.items():
    epoch_time = time_per_batch * batches_per_epoch
    total_time = epoch_time * num_epochs
    print(f"\n{name}:")
    print(f"  Time per epoch: {epoch_time/60:.1f} minutes")
    print(f"  Total training time: {total_time/3600:.1f} hours")

print("\n" + "="*80)
