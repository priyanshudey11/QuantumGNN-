# CPU Usage Optimization Fix

## Problem
Your M4 CPU usage is low (20-30%) because:

1. **12 DataLoader workers** competing with quantum simulation for CPU cores
2. **PyTorch threading limited** to 8 threads (you have 12 cores)
3. **Large batch size (12288)** means infrequent CPU bursts
4. **Thread contention** - too many workers fighting for the same resources

## Solution

### Add this cell at the TOP of your notebook (before imports):

```python
# ========== CPU OPTIMIZATION FOR M4 ==========
import torch
import os

# Enable ALL 12 cores for PyTorch operations
torch.set_num_threads(12)
torch.set_num_interop_threads(12)

# Enable multi-threading for BLAS/LAPACK (used by PennyLane)
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['VECLIB_MAXIMUM_THREADS'] = '12'
os.environ['NUMEXPR_NUM_THREADS'] = '12'

print("✓ CPU optimization enabled: 12 threads for quantum simulation")
```

### Update your DataLoader configuration:

Find this section in your notebook:
```python
EFFECTIVE_WORKERS = max(CPU_COUNT, int(CPU_COUNT * 0.95))  # Currently 12
BATCH_SIZE = max(256, int(BATCH_SIZE * 1.5))  # Currently 12288
```

**Replace with:**
```python
# ⚡ OPTIMIZED FOR QUANTUM SIMULATION
EFFECTIVE_WORKERS = 3   # Reduced: leave 9 cores for quantum circuits
BATCH_SIZE = 2048       # Reduced: more frequent CPU utilization
EFFECTIVE_PREFETCH = 4  # Keep this
```

## Why This Works

| Setting | Before | After | Reason |
|---------|--------|-------|--------|
| DataLoader workers | 12 | 3 | Reduce competition for CPU cores |
| PyTorch threads | 8 | 12 | Use ALL M4 cores for quantum simulation |
| Batch size | 12288 | 2048 | More frequent quantum circuit evaluations |
| OMP threads | default | 12 | Enable parallel BLAS operations |

## Expected Results

- **Before**: 20-30% CPU usage, threads mostly idle
- **After**: 70-90% CPU usage, cores fully utilized
- **Training speed**: 2-3x faster per epoch

## Visual Explanation

**Before (inefficient):**
```
12 cores: [W][W][W][W][W][W][W][W][W][W][W][W]
          ↑ All workers, fighting for resources
          ↑ Quantum simulation waits for available cores
          Result: 20-30% utilization (threads blocking each other)
```

**After (optimized):**
```
12 cores: [W][W][W][Q][Q][Q][Q][Q][Q][Q][Q][Q]
          ↑workers ↑ quantum simulation gets 9 cores
          Result: 70-90% utilization (efficient parallelism)
```

## Quick Test

After making changes, run one training batch and check Activity Monitor:
1. Open Activity Monitor (Command+Space → "Activity Monitor")
2. Click "CPU" tab
3. Look for `Python` process
4. Should see 70-90% CPU usage (not 20-30%)

## Implementation Steps

1. **Add CPU optimization cell** at top of notebook (before cell #1)
2. **Modify cell #10** (DataLoader configuration):
   - Change `EFFECTIVE_WORKERS = 12` → `3`
   - Change `BATCH_SIZE = 12288` → `2048`
3. **Restart kernel** and run from the beginning
4. **Monitor CPU usage** in Activity Monitor

That's it! Your CPU should now be properly utilized.
