# CPU Bottleneck Fixes Applied

## Problem Observed
- **CPU:** 100% utilization (bottleneck)
- **GPU:** 15% utilization (underutilized)
- **Cause:** Data loading can't keep up with fast `lightning.gpu` quantum circuit execution

---

## Solutions Applied

### 1. ⚡ Increased Batch Size: 512 → 1024
**Why:** Reduces the number of batch loads per epoch
- Fewer data loading operations
- More computation per GPU call
- Better amortization of data transfer overhead

### 2. 🔄 More Workers: 8 → 12
**Why:** More parallel CPU processes loading data
- Can utilize all CPU cores
- Multiple batches prepared simultaneously

### 3. 📦 Added Prefetch Factor: 4
**Why:** Pre-loads batches before GPU needs them
- Each of 12 workers prefetches 4 batches = 48 batches ready
- GPU never waits for data
- Smoother pipeline

### 4. ⚡ Non-blocking GPU Transfer
**Why:** CPU doesn't wait for GPU transfer to complete
```python
x.to(device, non_blocking=True)
```
- CPU can prepare next batch while current batch transfers
- Overlaps computation and data movement

### 5. 🗑️ Faster Gradient Zeroing
**Why:** `set_to_none=True` is faster than zeroing
```python
optimizer.zero_grad(set_to_none=True)
```
- Deallocates memory instead of writing zeros
- Reduces memory bandwidth usage

---

## Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Batch Size** | 512 | 1024 | 2x larger |
| **Workers** | 8 | 12 | 50% more |
| **Prefetch Batches** | 0 | 48 | ∞ better |
| **CPU Usage** | 100% | 80-90% | Less saturated |
| **GPU Usage** | 15% | 40-70% | 2.6-4.6x higher |
| **Batches/epoch** | ~382 | ~191 | 2x fewer |

---

## Why This Works

The quantum circuit on `lightning.gpu` is **very fast**, so:
1. GPU finishes a batch quickly
2. GPU waits for CPU to prepare next batch ← **BOTTLENECK**
3. CPU is busy loading, collating, transferring data

**Solution:** Give CPU a head start by prefetching many batches in parallel.

---

## Monitoring

Watch these metrics during training:

```bash
# Terminal 1: GPU utilization
watch -n 1 nvidia-smi

# Terminal 2: CPU utilization
htop

# Terminal 3: Training
jupyter notebook
```

**Expected:**
- GPU utilization: 40-70% (up from 15%)
- CPU usage: 80-90% (down from 100%)
- Faster epochs (fewer batches to process)

---

## Additional Optimization Ideas (if still bottlenecked)

### If CPU still at 100%:
1. **Cache preprocessed data** to disk (pickle)
2. **Simplify collate_fn** (profile with cProfile)
3. **Use torch.compile()** for model (PyTorch 2.0+)

### If GPU still low:
1. **Increase batch size** further (1024 → 2048)
2. **Remove `.cpu()` calls** in training loop
3. **Use mixed precision training** (though not compatible with quantum circuits)

### If memory errors:
1. **Reduce batch size** (1024 → 768)
2. **Reduce workers** (12 → 10)
3. **Reduce prefetch factor** (4 → 2)

---

## Configuration Summary

```python
# Current optimized settings
BATCH_SIZE = 1024
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
NON_BLOCKING_TRANSFER = True
```

These settings are tuned for:
- RTX 3080 (12GB VRAM)
- Multi-core CPU
- Fast quantum simulation on lightning.gpu
- Large dataset (244k interactions)

---

## Status: ✅ Applied

All optimizations have been implemented in the notebook. Just restart and run!
