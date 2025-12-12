"""
Profile exactly where time is spent in ONE training batch
This will tell us if it's:
1. Data loading (should be fast)
2. Classical network (should be fast)
3. Quantum circuit (expected to be slow)
"""

import torch
import numpy as np
import time
import os
import psutil
import threading
from datetime import datetime

# Enable CPU optimization
torch.set_num_threads(12)
os.environ['OMP_NUM_THREADS'] = '12'

# Import your code
from ligand_pocket_qgnn.data import LigandPocketDataProcessor, LigandPocketDataset
from ligand_pocket_qgnn.model import LigandPocketQGNN
from torch.utils.data import DataLoader

# CPU monitoring thread
cpu_samples = []
monitoring = False

def monitor_cpu():
    """Background thread to monitor CPU usage"""
    global cpu_samples, monitoring
    while monitoring:
        cpu_samples.append({
            'time': time.time(),
            'total': psutil.cpu_percent(interval=0.1),
            'per_core': psutil.cpu_percent(interval=0, percpu=True)
        })
        time.sleep(0.1)

print(f"{'='*70}")
print(f"🔬 PROFILING ONE TRAINING BATCH")
print(f"{'='*70}\n")

# Setup
DATA_DIR = "/Users/priyanshudey/Code/Qunatum copy/othercode/data"
BATCH_SIZE = 128  # Small batch for testing
SEED = 42069

print("1. Loading data (small sample)...")
processor = LigandPocketDataProcessor(DATA_DIR, seed=SEED)
processor.load_data(max_samples=1000)  # Small dataset
interactions = processor.get_dataset()[:200]  # Only 200 samples

print(f"   ✓ Loaded {len(interactions)} interactions\n")

# Create dataset
from sklearn.model_selection import train_test_split
train_ints, _ = train_test_split(interactions, test_size=0.2, random_state=SEED)
train_dataset = LigandPocketDataset(processor, train_ints)

# Collate function
def optimized_collate_fn(batch):
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

# Create loader with minimal workers
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=optimized_collate_fn,
    num_workers=0,  # No workers - run in main thread for clearer profiling
    pin_memory=False
)

print("2. Creating quantum model...")
sample_ligand = processor.ligands[interactions[0].ligand_id]
sample_pocket = processor.pockets[interactions[0].pocket_id]
ligand_dim = sample_ligand.atom_features.shape[1]
pocket_dim = sample_pocket.to_vector().shape[0]

model = LigandPocketQGNN(
    ligand_in_dim=ligand_dim,
    pocket_in_dim=pocket_dim,
    hidden_dim=64,
    n_qubits=6,
    n_qlayers=2,
    use_quantum=True,
    quantum_device='lightning.qubit'
)

print(f"   ✓ Model created\n")

print("3. Getting one batch...")
batch_iter = iter(train_loader)
x_batch, edge_index_batch, batch_vec, pocket_batch, labels = next(batch_iter)

print(f"   Batch size: {len(labels)}")
print(f"   Total nodes: {x_batch.shape[0]}")
print(f"   Total edges: {edge_index_batch.shape[1]}\n")

print(f"{'='*70}")
print(f"DETAILED PROFILING (with CPU monitoring)")
print(f"{'='*70}\n")

# Start CPU monitoring
cpu_samples = []
monitoring = True
monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
monitor_thread.start()

# Profile each component
results = {}

# 1. Ligand encoding
print("Step 1: Ligand Encoding (GNN)...")
start = time.time()
start_cpu = time.time()
with torch.no_grad():
    h_ligand = model.ligand_encoder(x_batch, edge_index_batch, batch_vec)
results['ligand_encoding'] = time.time() - start
print(f"   Time: {results['ligand_encoding']*1000:.2f}ms")
print(f"   Output shape: {h_ligand.shape}\n")

# 2. Pocket encoding
print("Step 2: Pocket Encoding (MLP)...")
start = time.time()
with torch.no_grad():
    h_pocket = model.pocket_encoder(pocket_batch)
results['pocket_encoding'] = time.time() - start
print(f"   Time: {results['pocket_encoding']*1000:.2f}ms")
print(f"   Output shape: {h_pocket.shape}\n")

# 3. Combine
print("Step 3: Combining features...")
start = time.time()
combined = torch.cat([h_ligand, h_pocket], dim=1)
combined = torch.tanh(combined) * torch.pi
results['combining'] = time.time() - start
print(f"   Time: {results['combining']*1000:.2f}ms")
print(f"   Combined shape: {combined.shape}\n")

# 4. Quantum circuit (THE BOTTLENECK)
print(f"Step 4: Quantum Circuit Evaluation...")
print(f"   Processing {len(combined)} samples through quantum circuit...")
print(f"   ⚠️  THIS WILL BE SLOW - WATCH YOUR CPU MONITOR!\n")

start = time.time()
cpu_before = [psutil.cpu_percent(interval=0, percpu=True)]

with torch.no_grad():
    quantum_output = model.interaction(combined)

results['quantum_circuit'] = time.time() - start
cpu_after = [psutil.cpu_percent(interval=0, percpu=True)]

print(f"   Time: {results['quantum_circuit']*1000:.2f}ms")
print(f"   Time per sample: {results['quantum_circuit']/len(combined)*1000:.2f}ms")
print(f"   Output shape: {quantum_output.shape}\n")

# Stop monitoring
monitoring = False
time.sleep(0.2)  # Let monitor thread finish

# 5. Full forward pass
print("Step 5: Full Forward Pass (end-to-end)...")
start = time.time()
with torch.no_grad():
    output = model(x_batch, edge_index_batch, batch_vec, pocket_batch)
results['full_forward'] = time.time() - start
print(f"   Time: {results['full_forward']*1000:.2f}ms\n")

print(f"{'='*70}")
print(f"TIMING BREAKDOWN")
print(f"{'='*70}\n")

total_time = sum(results.values())
for step, duration in results.items():
    pct = (duration / total_time) * 100
    bar = '█' * int(pct / 2)
    print(f"{step:<20} {duration*1000:>8.2f}ms  {pct:>5.1f}%  {bar}")

print(f"\n{'Total':<20} {total_time*1000:>8.2f}ms")

# CPU analysis
if cpu_samples:
    print(f"\n{'='*70}")
    print(f"CPU USAGE DURING QUANTUM EXECUTION")
    print(f"{'='*70}\n")

    # Get samples during quantum execution
    quantum_start = start
    quantum_samples = [s for s in cpu_samples
                      if s['time'] >= quantum_start and
                      s['time'] <= quantum_start + results['quantum_circuit']]

    if quantum_samples:
        avg_total = np.mean([s['total'] for s in quantum_samples])
        avg_per_core = np.mean([s['per_core'] for s in quantum_samples], axis=0)

        print(f"Average total CPU: {avg_total:.1f}%")
        print(f"\nPer-core usage:")
        for i, avg in enumerate(avg_per_core[:12]):
            bar = '█' * int(avg / 5)
            print(f"  Core {i:2d}: {avg:>5.1f}%  {bar}")

print(f"\n{'='*70}")
print(f"CONCLUSION")
print(f"{'='*70}")

quantum_pct = (results['quantum_circuit'] / total_time) * 100
print(f"\nQuantum circuit takes {quantum_pct:.1f}% of total time")

if quantum_pct > 80:
    print(f"✓ As expected - quantum simulation is the bottleneck")
    print(f"✓ Classical components (GNN, MLP) are fast")
    print(f"✓ This is NORMAL for PennyLane quantum simulation")
else:
    print(f"⚠️  Unexpected - quantum is not the main bottleneck")
    print(f"⚠️  Check data loading or classical network performance")

print(f"\n{'='*70}\n")
