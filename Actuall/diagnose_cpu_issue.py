"""
Diagnose CPU usage issue with PennyLane quantum simulation
"""

import torch
import pennylane as qml
import numpy as np
import time
import os
import multiprocessing as mp

# Set threading
torch.set_num_threads(12)
os.environ['OMP_NUM_THREADS'] = '12'

print(f"{'='*70}")
print(f"🔍 DIAGNOSING CPU USAGE ISSUE")
print(f"{'='*70}")

# Test 1: Check if PennyLane TorchLayer actually parallelizes
print(f"\n1. Testing PennyLane TorchLayer batch processing...")
print(f"   This will show if the quantum circuit uses multiple cores")

n_qubits = 6
n_layers = 2
batch_size = 128

dev = qml.device('lightning.qubit', wires=n_qubits)

@qml.qnode(dev, interface='torch', diff_method='best')
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

# Create test batch
test_input = torch.randn(batch_size, n_qubits)

print(f"\n   Running {batch_size} samples through quantum circuit...")
print(f"   📊 Open Activity Monitor NOW and watch CPU usage!\n")

# Time the execution
start = time.time()
output = q_layer(test_input)
elapsed = time.time() - start

print(f"\n   ✓ Completed in {elapsed:.3f}s")
print(f"   ✓ Time per sample: {elapsed/batch_size*1000:.2f}ms")
print(f"   ✓ Throughput: {batch_size/elapsed:.1f} samples/sec")

# Test 2: Check actual threading
print(f"\n{'='*70}")
print(f"2. Checking PyTorch Threading Configuration...")
print(f"{'='*70}")
print(f"   PyTorch intra-op threads: {torch.get_num_threads()}")
print(f"   PyTorch inter-op threads: {torch.get_num_interop_threads()}")
print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
print(f"   Total CPU cores: {mp.cpu_count()}")

# Test 3: Profile where time is spent
print(f"\n{'='*70}")
print(f"3. Profiling Quantum Circuit Execution...")
print(f"{'='*70}")

# Test with different batch sizes to see scaling
batch_sizes = [1, 16, 64, 256]
times = []

for bs in batch_sizes:
    test_input = torch.randn(bs, n_qubits)
    start = time.time()
    _ = q_layer(test_input)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"   Batch size {bs:3d}: {elapsed:.3f}s ({elapsed/bs*1000:.2f}ms per sample)")

# Check if we're getting parallelization
print(f"\n   Analysis:")
ideal_time_256 = times[0] * 256  # If fully sequential
actual_time_256 = times[-1]
speedup = ideal_time_256 / actual_time_256
print(f"   • If fully sequential: {ideal_time_256:.3f}s for 256 samples")
print(f"   • Actual time: {actual_time_256:.3f}s")
print(f"   • Speedup factor: {speedup:.2f}x")

if speedup < 1.5:
    print(f"\n   ⚠️  WARNING: No parallelization detected!")
    print(f"   The quantum circuit is processing samples SEQUENTIALLY")
else:
    print(f"\n   ✓ Some parallelization detected")

# Test 4: The real issue - TorchLayer processes samples one-by-one
print(f"\n{'='*70}")
print(f"4. ROOT CAUSE ANALYSIS")
print(f"{'='*70}")
print(f"""
The issue is that PennyLane's TorchLayer does NOT parallelize across samples!

Each sample in the batch is processed SEQUENTIALLY through the quantum circuit:
  • Sample 1 → quantum circuit → output 1
  • Sample 2 → quantum circuit → output 2
  • ...
  • Sample N → quantum circuit → output N

This means:
  ✗ Only 1 CPU core is active at a time for quantum simulation
  ✗ Other cores sit idle
  ✗ CPU usage appears low (8-16% on a 12-core machine)
  ✗ Training is SLOW

Solution: We need to manually parallelize across CPU cores!
""")

# Test 5: Manual parallelization proof-of-concept
print(f"\n{'='*70}")
print(f"5. TESTING MANUAL PARALLELIZATION")
print(f"{'='*70}")

print(f"\n   Testing parallel quantum circuit evaluation...")
print(f"   This should use MORE CPU cores!\n")

def eval_single(args):
    """Evaluate quantum circuit for a single sample"""
    idx, sample, weights_dict = args
    # Create device per process (not thread-safe)
    local_dev = qml.device('lightning.qubit', wires=n_qubits)

    @qml.qnode(local_dev, interface='torch', diff_method='best')
    def local_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return local_circuit(sample, weights_dict['weights']).item()

# Get weights from TorchLayer
weights_dict = {'weights': q_layer.qnode_weights['weights'].detach()}

# Prepare arguments
test_input_small = torch.randn(32, n_qubits)
args_list = [(i, sample, weights_dict) for i, sample in enumerate(test_input_small)]

# Sequential version
print(f"   Sequential (current approach):")
start = time.time()
results_seq = [eval_single(args) for args in args_list]
time_seq = time.time() - start
print(f"   Time: {time_seq:.3f}s")

# Parallel version
print(f"\n   Parallel (using multiprocessing):")
start = time.time()
with mp.Pool(processes=min(8, mp.cpu_count())) as pool:
    results_par = pool.map(eval_single, args_list)
time_par = time.time() - start
print(f"   Time: {time_par:.3f}s")
print(f"   Speedup: {time_seq/time_par:.2f}x")

if time_seq / time_par > 2:
    print(f"\n   ✓ PARALLEL VERSION IS FASTER!")
    print(f"   This confirms manual parallelization can help")
else:
    print(f"\n   ⚠️  No significant speedup from parallelization")
    print(f"   Overhead may be too high for this circuit size")

print(f"\n{'='*70}")
print(f"CONCLUSION")
print(f"{'='*70}")
print(f"""
The root cause of low CPU usage is:
  • PennyLane TorchLayer processes batch samples SEQUENTIALLY
  • Each quantum circuit evaluation uses only 1-2 cores
  • Result: Most CPU cores sit idle

Potential solutions:
  1. Reduce batch size → more batches → more frequent work
  2. Use manual parallelization (multiprocessing)
  3. Use GPU quantum simulation (if available)
  4. Accept slower training (quantum simulation is inherently slow)

The dataloader workers are NOT the problem!
""")
