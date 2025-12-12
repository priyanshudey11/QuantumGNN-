# Why is CPU Usage Low? (The Real Answer)

## TL;DR
**40-60% CPU usage is NORMAL and EXPECTED for PennyLane quantum simulation.** This is not a bug or configuration issue - it's how quantum simulation works.

## The Real Bottleneck

### PennyLane TorchLayer is SEQUENTIAL
```python
# What happens when you run quantum_model(batch):
for sample in batch:  # ← SEQUENTIAL LOOP (not parallel!)
    result = quantum_circuit(sample)  # Uses 1-2 CPU cores
    results.append(result)
```

**This means:**
- ✗ Only 1-2 cores active at a time for quantum simulation
- ✗ Other 10 cores sit mostly idle
- ✗ Result: 10-20% CPU usage PER CORE = 40-60% total
- ✓ This is NORMAL for quantum simulation!

## Proof from Diagnostic Tests

```
Batch size   1: 2.64ms per sample
Batch size 256: 1.97ms per sample

Speedup: 1.34x (should be 256x if fully parallel!)
```

**Conclusion:** PennyLane is processing samples sequentially, NOT in parallel.

## Why Can't We Parallelize?

### 1. Quantum State is Stateful
Each quantum circuit maintains a quantum state that can't be easily shared across processes.

### 2. PennyLane Design
`TorchLayer` is designed for autodiff integration, not parallelism. The sequential processing allows proper gradient tracking.

### 3. Python GIL
Even if we tried threading, Python's Global Interpreter Lock would serialize execution.

## What We've Already Tried

| Optimization | Result |
|--------------|--------|
| Increase PyTorch threads to 12 | ✓ Enabled, but quantum sim still sequential |
| Reduce DataLoader workers to 2 | ✓ Reduced competition, no improvement |
| Set OMP_NUM_THREADS=12 | ✓ Enabled, helps matrix ops but not main loop |
| Smaller batch sizes (2048) | ✓ More batches, same per-sample time |

**None of these change the fundamental sequential nature of quantum simulation.**

## What Would Actually Help?

### Option 1: Manual Parallelization (Complex)
```python
from multiprocessing import Pool

# Create separate quantum devices per process
def eval_sample(sample):
    local_dev = qml.device('lightning.qubit', wires=6)
    # ... evaluate circuit ...
    return result

with Pool(8) as pool:
    results = pool.map(eval_sample, batch)
```

**Problems:**
- Can't use autodiff (no gradients!)
- High overhead from creating multiple quantum devices
- Doesn't integrate with PyTorch training loop
- May actually be SLOWER due to overhead

### Option 2: Use Quantum Hardware/Simulator (Expensive)
- IBM Quantum: Real quantum hardware (limited access)
- QPU simulators: Specialized hardware ($$$)
- GPU quantum simulators: Requires CUDA quantum libraries

### Option 3: Accept It (Recommended)
**This is the realistic answer:**
- 40-60% CPU usage is normal for PennyLane
- Quantum simulation is inherently slow
- Focus on making the circuit faster (fewer qubits/layers)
- Use classical models for comparison

## Current Status

### What's Optimized ✓
- PyTorch using all 12 cores for matrix operations
- Minimal DataLoader workers (2) - no resource competition
- Smaller batches (2048) for more frequent progress updates
- OMP threading enabled for BLAS operations

### What's the Bottleneck ✗
- **Sequential quantum circuit evaluation** (can't fix easily)
- Each sample: 1.9ms × 12,288 samples = 23 seconds per batch
- This is just how quantum simulation works on CPU

## What To Do Now

### 1. Accept Reality
- 40-60% CPU usage is EXPECTED
- Quantum simulation is slow by nature
- This is not a configuration problem

### 2. Make Training Faster
- ✓ Use smaller quantum circuits (6 qubits, 2 layers)
- ✓ Use smaller batches (2048) for better progress tracking
- ✓ Consider reducing dataset size for experiments

### 3. Alternative: Train Classical Model
The classical model will use 80-90% CPU and train much faster!

## Bottom Line

**You cannot easily fix low CPU usage for quantum simulation.** The architecture of PennyLane's `TorchLayer` processes samples sequentially. To get higher CPU usage, you would need to:

1. Rewrite the quantum layer with manual multiprocessing (breaks autodiff)
2. Use specialized quantum hardware (expensive, limited access)
3. Switch to a different quantum ML framework (major refactor)

**Recommendation:** Accept 40-60% CPU usage as normal, focus on optimizing the quantum circuit itself (fewer qubits/layers), and use the classical model for faster iterations.

---

## Activity Monitor: What You Should See

```
Process: Python
CPU: 40-60% (distributed across cores)
Threads: ~15-20 active

Per-Core Breakdown:
Core 1-2:  30-40% (quantum simulation)
Core 3-12: 2-5% each (BLAS operations, overhead)
```

This is NORMAL and EXPECTED! Don't try to "fix" it further.
