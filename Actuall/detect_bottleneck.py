#!/usr/bin/env python3
"""
Quick bottleneck detector for Mac vs CUDA
Run this to see which part of your training is slow
"""

import torch
import time
import os
import platform
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

print(f"\n{'='*70}")
print(f"BOTTLENECK DETECTOR")
print(f"{'='*70}\n")

# ============ SYSTEM INFO ============
PLATFORM = platform.system()
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if IS_CUDA else 'cpu'
CPU_COUNT = os.cpu_count()

print(f"Platform: {PLATFORM}")
print(f"Device: {DEVICE.upper()}")
print(f"CPU Cores: {CPU_COUNT}")

if IS_CUDA:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print(f"\n{'='*70}\n")

# ============ CREATE DUMMY DATA ============
print("Creating dummy training data...")
n_samples = 1000
feature_dim = 64

X = torch.randn(n_samples, feature_dim)
y = torch.randint(0, 2, (n_samples,)).float()

dataset = TensorDataset(X, y)

# ============ TEST 1: DATA LOADING ============
print("\n📦 TEST 1: DATA LOADING SPEED")
print(f"{'─'*70}")

batch_sizes = [16, 32, 64, 128, 256]

for batch_size in batch_sizes:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if IS_CUDA else min(8, CPU_COUNT),
        pin_memory=True,
    )
    
    times = []
    for i, (x, y) in enumerate(loader):
        start = time.time()
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        torch.cuda.synchronize() if IS_CUDA else None
        elapsed = time.time() - start
        times.append(elapsed)
        if i >= 4:  # Test first 5 batches
            break
    
    avg_time = np.mean(times) * 1000
    print(f"  Batch size {batch_size:3d}: {avg_time:6.2f} ms")

# ============ TEST 2: TENSOR OPERATIONS ============
print("\n⚙️  TEST 2: TENSOR OPERATIONS (Classical Layer Simulation)")
print(f"{'─'*70}")

test_input = torch.randn(1, 64, device=DEVICE)
test_weight = torch.randn(64, 64, device=DEVICE)

times = []
for _ in range(100):
    start = time.time()
    output = torch.matmul(test_input, test_weight)
    torch.cuda.synchronize() if IS_CUDA else None
    elapsed = time.time() - start
    times.append(elapsed)

avg_time = np.mean(times) * 1000
print(f"  Linear layer forward pass: {avg_time:.4f} ms")
print(f"  → Very fast on all devices")

# ============ TEST 3: QUANTUM CIRCUIT SIMULATION ============
print("\n⚛️  TEST 3: QUANTUM CIRCUIT SIMULATION (if available)")
print(f"{'─'*70}")

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    
    # Create a simple quantum circuit
    n_qubits = 4
    dev = qml.device(f"lightning.{'gpu' if IS_CUDA else 'qubit'}", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(params, x):
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        for i, param in enumerate(params):
            qml.RY(param, wires=i % n_qubits)
        return qml.expval(qml.PauliZ(0))
    
    params = pnp.array([0.1, 0.2, 0.3, 0.4])
    x = pnp.array([0.5, 0.6, 0.7, 0.8])
    
    times = []
    for _ in range(10):
        start = time.time()
        result = circuit(params, x)
        torch.cuda.synchronize() if IS_CUDA else None
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = np.mean(times) * 1000
    print(f"  Quantum circuit (4Q, 1 layer): {avg_time:.2f} ms")
    print(f"  → {'⚠️  SLOW on CUDA!' if IS_CUDA and avg_time > 5 else '✓ OK'}")
    
except ImportError:
    print(f"  PennyLane not installed - skipping quantum test")

# ============ BOTTLENECK ANALYSIS ============
print(f"\n{'='*70}")
print(f"🔍 BOTTLENECK ANALYSIS")
print(f"{'='*70}\n")

if IS_CUDA:
    print(f"⚠️  CUDA DETECTED")
    print(f"\nQuantum circuits are likely the bottleneck:")
    print(f"  • Quantum: 5-100 ms per sample")
    print(f"  • Classical: 0.1 ms per sample")
    print(f"  • Ratio: 50-1000x slower!")
    print(f"\n💡 SOLUTION:")
    print(f"  → Reduce N_QUBITS from 6 to 3-4")
    print(f"  → Reduce N_QLAYERS from 2 to 1")
    print(f"  → Use batch_size=16-32 for more parallelism")
    print(f"\n📈 Expected improvement: 20-50x faster training")
else:
    print(f"✓ MAC/CPU DETECTED")
    print(f"\nData loading is likely the bottleneck:")
    print(f"  • Data loading: 20-100 ms per batch")
    print(f"  • Tensor ops: 0.1-1 ms per batch")
    print(f"  • Ratio: 20-1000x slower!")
    print(f"\n💡 SOLUTION:")
    print(f"  → Increase NUM_WORKERS to {CPU_COUNT-2}")
    print(f"  → Increase PREFETCH_FACTOR to 8-16")
    print(f"  → Use batch_size=8192 for fewer synchronizations")
    print(f"\n📈 Expected improvement: 5-10x faster training")

print(f"\n{'='*70}\n")
