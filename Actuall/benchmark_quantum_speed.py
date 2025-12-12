#!/usr/bin/env python3
"""
Benchmark script to diagnose quantum circuit performance bottlenecks.
Shows execution times for different batch sizes and optimization settings.
"""

import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
import time
from datetime import datetime

def benchmark_quantum_circuits():
    print(f"\n{'='*80}")
    print(f"🔬 QUANTUM CIRCUIT PERFORMANCE BENCHMARK")
    print(f"{'='*80}\n")
    
    # Hardware info
    n_qubits = 8
    n_layers = 4
    batch_sizes = [1, 4, 8, 16, 32, 64]
    
    print(f"System Configuration:")
    print(f"  Qubits: {n_qubits}")
    print(f"  Layers: {n_layers}")
    print(f"  Trainable params: {n_layers * n_qubits * 3}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"\n{'='*80}\n")
    
    # Test different devices
    devices_to_test = []
    
    # Try GPU
    try:
        dev_gpu = qml.device('lightning.gpu', wires=n_qubits)
        devices_to_test.append(('lightning.gpu', dev_gpu))
        print("✓ lightning.gpu available")
    except Exception as e:
        print(f"✗ lightning.gpu not available: {e}")
    
    # Try default
    try:
        dev_cpu = qml.device('lightning.qubit', wires=n_qubits)
        devices_to_test.append(('lightning.qubit', dev_cpu))
        print("✓ lightning.qubit available")
    except Exception as e:
        print(f"✗ lightning.qubit not available: {e}")
    
    print(f"\n{'='*80}\n")
    
    # Benchmark each device and batch size
    for device_name, device in devices_to_test:
        print(f"\n{'='*80}")
        print(f"Testing: {device_name}")
        print(f"{'='*80}\n")
        
        # Define circuit
        @qml.qnode(device, interface='torch')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))
        
        # Create TorchLayer
        q_layer = qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 3)})
        
        print(f"{'Batch Size':<15} {'Time (ms)':<15} {'Per Sample (ms)':<20} {'Samples/sec':<15}")
        print(f"{'-'*65}")
        
        results = {}
        for batch_size in batch_sizes:
            # Create random batch
            inputs = torch.randn(batch_size, n_qubits)
            
            # Warm up
            _ = q_layer(inputs)
            
            # Benchmark
            start = time.time()
            for _ in range(3):  # 3 iterations
                output = q_layer(inputs)
            end = time.time()
            
            total_time = (end - start) * 1000 / 3  # Convert to ms, average
            per_sample = total_time / batch_size
            samples_per_sec = 1000 / per_sample
            
            results[batch_size] = {
                'total_ms': total_time,
                'per_sample_ms': per_sample,
                'samples_per_sec': samples_per_sec
            }
            
            print(f"{batch_size:<15} {total_time:<15.2f} {per_sample:<20.4f} {samples_per_sec:<15.1f}")
        
        # Calculate scaling
        print(f"\n{'Scaling Analysis':}")
        speedup = results[batch_sizes[-1]]['per_sample_ms'] / results[batch_sizes[0]]['per_sample_ms']
        print(f"  Speedup (B={batch_sizes[-1]} vs B={batch_sizes[0]}): {speedup:.2f}x")
        
        # Optimal batch size (lowest per-sample time)
        best_batch = min(results.keys(), key=lambda b: results[b]['per_sample_ms'])
        print(f"  Optimal batch size: {best_batch} (per-sample time: {results[best_batch]['per_sample_ms']:.4f}ms)")
        
        print(f"\n")

def benchmark_classical_vs_quantum():
    """Compare classical MLP vs quantum circuit."""
    print(f"\n{'='*80}")
    print(f"📊 CLASSICAL vs QUANTUM COMPARISON")
    print(f"{'='*80}\n")
    
    n_qubits = 8
    batch_sizes = [1, 8, 16, 32]
    
    # Classical MLP
    classical = nn.Sequential(
        nn.Linear(n_qubits, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    
    # Quantum
    dev_quantum = qml.device('lightning.qubit', wires=n_qubits)
    
    @qml.qnode(dev_quantum, interface='torch')
    def q_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits, 3))
        return qml.expval(qml.PauliZ(0))
    
    q_layer = qml.qnn.TorchLayer(q_circuit, {"weights": (4, n_qubits, 3)})
    
    print(f"{'Batch':<10} {'Classical (ms)':<18} {'Quantum (ms)':<18} {'Ratio (Q/C)':<15}")
    print(f"{'-'*60}")
    
    for batch_size in batch_sizes:
        inputs = torch.randn(batch_size, n_qubits)
        
        # Classical timing
        start = time.time()
        for _ in range(10):
            _ = classical(inputs)
        classical_time = (time.time() - start) * 1000 / 10
        
        # Quantum timing
        start = time.time()
        for _ in range(10):
            _ = q_layer(inputs)
        quantum_time = (time.time() - start) * 1000 / 10
        
        ratio = quantum_time / classical_time
        
        print(f"{batch_size:<10} {classical_time:<18.4f} {quantum_time:<18.4f} {ratio:<15.1f}x")
    
    print(f"\n💡 Insight: Quantum circuits are typically 10-100x slower than classical MLPs")
    print(f"   Use quantum only where quantum advantage is expected.\n")

def recommendations():
    print(f"\n{'='*80}")
    print(f"⚡ OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    print(f"""
1. BATCH SIZE OPTIMIZATION
   - Use larger batch sizes (32+) to amortize quantum overhead
   - Each sample has fixed quantum setup cost
   - Larger batches reduce per-sample overhead
   
2. QUANTUM DEVICE SELECTION
   - Try 'lightning.gpu' if available (2-5x faster than CPU)
   - Falls back to 'lightning.qubit' if needed
   - GPU acceleration is critical for M4

3. CIRCUIT OPTIMIZATION
   - Reduce n_qubits if possible (fewer qubits = faster)
   - Reduce n_qlayers to minimum that works
   - Use single measurement (not multiple)

4. TRAINING STRATEGY
   - Consider hybrid: Use quantum only for critical interactions
   - Pre-compute classical features in parallel
   - Use gradient caching to avoid redundant evaluations

5. EXPECTED PERFORMANCE
   - Classical MLP: ~0.01-0.1ms per sample
   - Quantum VQC: ~1-10ms per sample (10-100x slower)
   - With batch_size=32: ~300-3000ms per batch

6. FURTHER OPTIMIZATION
   - Reduce qubits from 8 to 6 (less overhead)
   - Reduce layers from 4 to 2 (faster ansatz)
   - Use parameter-shift (gradient) instead of finite diff
   - Enable gradient caching in TorchLayer
""")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    benchmark_quantum_circuits()
    benchmark_classical_vs_quantum()
    recommendations()
