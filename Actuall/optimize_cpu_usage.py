"""
CPU Optimization Configuration for M4 Quantum Training

This script optimizes threading for quantum simulation on Apple Silicon M4.
Run this BEFORE your training notebook.
"""

import os
import torch
import multiprocessing as mp

def optimize_for_m4_quantum():
    """
    Optimize CPU usage for M4 chip running PennyLane quantum simulations.

    Key insights:
    - M4 has 12 cores (likely 6 P-cores + 6 E-cores)
    - Quantum simulation is CPU-bound and benefits from multi-threading
    - DataLoader workers compete with quantum simulation for cores
    """

    cpu_count = mp.cpu_count()
    print(f"{'='*70}")
    print(f"🚀 M4 CPU OPTIMIZATION FOR QUANTUM TRAINING")
    print(f"{'='*70}")
    print(f"Total CPU cores: {cpu_count}")

    # STRATEGY 1: Maximize PyTorch threads for quantum simulation
    # The quantum circuit evaluation is the bottleneck, not data loading

    # Set PyTorch to use ALL cores for intra-op parallelism (matrix operations)
    torch.set_num_threads(cpu_count)

    # Set inter-op parallelism (operations running in parallel)
    torch.set_num_interop_threads(cpu_count)

    # Enable MKL/BLAS threading (for linear algebra operations)
    os.environ['OMP_NUM_THREADS'] = str(cpu_count)
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_count)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)

    print(f"\n✓ PyTorch intra-op threads: {torch.get_num_threads()}")
    print(f"✓ PyTorch inter-op threads: {torch.get_num_interop_threads()}")
    print(f"✓ OMP threads: {cpu_count}")

    # STRATEGY 2: Reduce DataLoader workers to avoid competition
    # Fewer workers = more cores available for quantum simulation
    recommended_workers = max(2, cpu_count // 4)  # Use 25% of cores for data loading

    print(f"\n{'='*70}")
    print(f"RECOMMENDED DATALOADER SETTINGS:")
    print(f"{'='*70}")
    print(f"num_workers: {recommended_workers} (down from {cpu_count})")
    print(f"prefetch_factor: 4")
    print(f"persistent_workers: True")
    print(f"\nRationale:")
    print(f"  • {recommended_workers} workers for data loading")
    print(f"  • {cpu_count - recommended_workers} cores free for quantum simulation")
    print(f"  • Reduces thread contention")
    print(f"  • Better CPU utilization (70-90% expected)")

    # STRATEGY 3: Enable PennyLane parallel execution
    print(f"\n{'='*70}")
    print(f"PENNYLANE OPTIMIZATION:")
    print(f"{'='*70}")
    print(f"✓ Set OMP_NUM_THREADS={cpu_count} (enables parallel matrix ops)")
    print(f"✓ lightning.qubit will use multiple cores automatically")
    print(f"✓ Batch execution through TorchLayer")

    # STRATEGY 4: Adjust batch size for better parallelization
    # Smaller batches = more frequent CPU utilization
    recommended_batch = 2048  # Down from 12288

    print(f"\n{'='*70}")
    print(f"RECOMMENDED BATCH SIZE:")
    print(f"{'='*70}")
    print(f"Batch size: {recommended_batch} (down from 12288)")
    print(f"\nRationale:")
    print(f"  • Smaller batches = more frequent quantum circuit evaluations")
    print(f"  • More opportunities for parallel execution")
    print(f"  • Better CPU core utilization")
    print(f"  • More batches per epoch = better progress tracking")

    print(f"\n{'='*70}")
    print(f"EXPECTED RESULTS:")
    print(f"{'='*70}")
    print(f"Before: 20-30% CPU usage (threads idle, workers competing)")
    print(f"After:  70-90% CPU usage (cores fully utilized)")
    print(f"{'='*70}\n")

    return {
        'num_workers': recommended_workers,
        'batch_size': recommended_batch,
        'num_threads': cpu_count
    }


if __name__ == "__main__":
    config = optimize_for_m4_quantum()

    print("\n🔧 Add this to your notebook BEFORE creating DataLoaders:\n")
    print(f"""
import torch
import os

# Optimize CPU usage
torch.set_num_threads({config['num_threads']})
torch.set_num_interop_threads({config['num_threads']})
os.environ['OMP_NUM_THREADS'] = '{config['num_threads']}'
os.environ['MKL_NUM_THREADS'] = '{config['num_threads']}'
os.environ['VECLIB_MAXIMUM_THREADS'] = '{config['num_threads']}'

# Update DataLoader settings
NUM_WORKERS = {config['num_workers']}  # Reduced from 12
BATCH_SIZE = {config['batch_size']}    # Reduced from 12288
PREFETCH_FACTOR = 4
""")
