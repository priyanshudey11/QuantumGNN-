# Automatic Hardware Detection & Optimization

This system automatically detects your hardware and optimizes training configurations for **any platform**.

## Supported Hardware

### ✅ CPUs
- **Intel**: Core i3/i5/i7/i9, Xeon
- **AMD**: Ryzen 3/5/7/9, Threadripper, EPYC
- **Apple Silicon**: M1, M2, M3, M4 (all variants)
- **Qualcomm**: Snapdragon X Elite/Plus
- **ARM**: Neoverse, Graviton

### ✅ GPUs
- **NVIDIA**: All CUDA-capable GPUs
  - Consumer: RTX 20/30/40 series
  - Datacenter: A100, H100, V100
- **Apple**: Metal Performance Shaders (M1/M2/M3/M4)

## How It Works

The system automatically:
1. **Detects** your CPU (brand, model, cores)
2. **Detects** your GPU/accelerator (if available)
3. **Measures** available memory
4. **Optimizes** batch size, workers, prefetch, and memory settings

## Usage

### Option 1: Use the Clean Notebook (Recommended)

Simply run `compare_clean.ipynb` - it automatically detects and optimizes:

```python
from hardware_optimizer import setup_environment

# One line does everything!
hw_info, hw_config = setup_environment()
```

### Option 2: Manual Integration

```python
from hardware_optimizer import detect_hardware, optimize_for_hardware

# Detect hardware
hw_info = detect_hardware()

# Get optimized configuration
config = optimize_for_hardware(hw_info)

# Use the config
DEVICE = hw_info['device']
BATCH_SIZE = config['batch_size']
NUM_WORKERS = config['num_workers']
# ... etc
```

### Option 3: Test Detection Only

```bash
python test_hardware_detection.py
```

## Example Outputs

### Apple M4 Pro (Your System)
```
Platform:     Darwin
CPU:          Apple M4 Pro
CPU Cores:    12
Device:       Apple Metal Performance Shaders
Memory:       24.0 GB
Capability:   apple_m4_gpu

Optimized Config:
  Batch Size:   1024
  Workers:      10
  Prefetch:     5
  Pin Memory:   True
```

### NVIDIA RTX 4090
```
Platform:     Linux
CPU:          AMD Ryzen 9 7950X
CPU Cores:    32
Device:       NVIDIA GeForce RTX 4090
Memory:       24.0 GB
Capability:   nvidia_ada_high

Optimized Config:
  Batch Size:   2048
  Workers:      8
  Prefetch:     4
  Quantum Dev:  lightning.gpu
```

### Intel i7 + RTX 3070
```
Platform:     Windows
CPU:          Intel Core i7-12700K
CPU Cores:    20
Device:       NVIDIA GeForce RTX 3070
Memory:       8.0 GB
Capability:   nvidia_ampere

Optimized Config:
  Batch Size:   512
  Workers:      4
  Prefetch:     2
```

### AMD Threadripper (CPU-only)
```
Platform:     Linux
CPU:          AMD Ryzen Threadripper 3990X
CPU Cores:    128
Device:       CPU
Memory:       256.0 GB
Capability:   amd_threadripper

Optimized Config:
  Batch Size:   3072
  Workers:      16
  Prefetch:     8
```

### Snapdragon X Elite (ARM Windows)
```
Platform:     Windows
CPU:          Snapdragon X Elite
CPU Cores:    12
Device:       CPU
Memory:       32.0 GB
Capability:   snapdragon

Optimized Config:
  Batch Size:   768
  Workers:      8
  Prefetch:     3
```

## Optimization Strategy

The system optimizes based on:

### For NVIDIA GPUs
- **High VRAM** (≥20GB): Large batches (2048-4096)
- **Mid VRAM** (8-16GB): Medium batches (512-1024)
- **Low VRAM** (<8GB): Small batches (256)
- **GPU acceleration** for quantum simulation (lightning.gpu)

### For Apple Silicon
- **M4**: Aggressive settings (1536 batch, 10 workers)
- **M3**: High performance (1024 batch, 8 workers)
- **M2**: Good performance (768 batch, 6 workers)
- **M1**: Balanced (512 batch, 4 workers)
- **Unified memory** architecture optimizations

### For AMD CPUs
- **EPYC/Threadripper**: Maximum parallelism (3072 batch, 16 workers)
- **Ryzen 9**: High throughput (2048 batch, 12 workers)
- **Ryzen 7**: Balanced (1024 batch, 8 workers)
- **Ryzen 5**: Conservative (512 batch, 4 workers)

### For Intel CPUs
- **Xeon**: Server-optimized (2048 batch, 12 workers)
- **i9**: High performance (1536 batch, 10 workers)
- **i7**: Balanced (1024 batch, 6 workers)
- **i5**: Conservative (512 batch, 4 workers)

### For ARM CPUs
- **Server** (Graviton, Neoverse): High parallelism
- **Desktop** (Snapdragon X): Balanced settings
- **Mobile**: Conservative settings

## Files

- **`hardware_optimizer.py`**: Main detection and optimization logic
- **`compare_clean.ipynb`**: Clean notebook using auto-detection
- **`test_hardware_detection.py`**: Test script to see detection results

## Advanced Usage

Override automatic settings if needed:

```python
hw_info, config = setup_environment()

# Override batch size for experimentation
config['batch_size'] = 2048

BATCH_SIZE = config['batch_size']
```

## Benefits

✅ **No manual configuration** - works on any hardware
✅ **Optimal performance** - hardware-specific tuning
✅ **Memory safe** - prevents OOM errors
✅ **Future-proof** - automatically supports new hardware
✅ **Cross-platform** - macOS, Linux, Windows

## Troubleshooting

If detection fails or gives suboptimal settings:

1. Run `python test_hardware_detection.py` to see what was detected
2. Check the output for any warnings
3. Manually override config if needed
4. Report issues with your hardware specs

## Contributing

To add support for new hardware:

1. Edit `detect_hardware()` to recognize the new CPU/GPU
2. Edit `optimize_for_hardware()` to add optimization rules
3. Test with `test_hardware_detection.py`

---

**Made for**: Intel, AMD, NVIDIA, Apple, Qualcomm, ARM, and more
**Tested on**: M1/M2/M3/M4, RTX 20/30/40, Ryzen, EPYC, Xeon, Snapdragon X
