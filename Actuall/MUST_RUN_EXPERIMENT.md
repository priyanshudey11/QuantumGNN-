# 🔬 MUST-RUN EXPERIMENT CONFIGURATION

## Experiment Setup

This experiment compares **Quantum** vs **Classical** Graph Neural Networks for ligand-pocket binding prediction with optimized parameters.

---

## ⭐ Quantum Circuit Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Qubits** | `8` | Increased from 6 (33% more quantum capacity) |
| **Depth (Layers)** | `4` | Increased from 2 (2x deeper circuit) |
| **Ansatz** | `StronglyEntanglingLayers` | Full entanglement between all qubits |
| **Device** | `lightning.gpu` | GPU-accelerated quantum simulation |
| **Fallback** | `lightning.qubit` | CPU fallback if GPU unavailable |

**Trainable Quantum Parameters:** `4 layers × 8 qubits × 3 rotations = 96 parameters`

---

## 🚀 Optimization Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Batch Size** | `512` | 4x larger than baseline (128) - maximizes GPU utilization |
| **Num Workers** | `8` | Parallel data loading on CPU |
| **Pin Memory** | `True` | Faster CPU→GPU transfer |
| **cuDNN Benchmark** | `True` | Optimized convolution algorithms |
| **Persistent Workers** | `True` | Reuse worker processes between epochs |

---

## 📊 Expected Performance

### Speed Improvements
- **Per-batch time:** ~1.2s → ~0.3s (4x faster)
- **Per-epoch time:** ~35 min → ~8 min (4.4x faster)
- **GPU utilization:** 5% → 40-60% (8-12x improvement)
- **GPU memory usage:** 1.8GB → 4-6GB

### Model Capacity
- **Quantum parameters:** 36 → 96 (2.67x more parameters)
- **Circuit depth:** 2 → 4 (2x deeper)
- **Qubit space:** 2^6 = 64 → 2^8 = 256 (4x larger Hilbert space)

---

## 🎯 Quantum Advantage Hypothesis

With 8 qubits and depth 4, the quantum model should be able to:

1. **Capture more complex interactions** between ligand and pocket features
2. **Exploit quantum entanglement** for non-linear feature correlations
3. **Utilize larger Hilbert space** (256-dimensional) for representation learning
4. **Benefit from GPU acceleration** via `lightning.gpu`

---

## 📝 Training Configuration

```python
# Model Parameters
N_QUBITS = 8
N_QLAYERS = 4
HIDDEN_DIM = 64
QUANTUM_DEVICE = 'lightning.gpu'

# Training Parameters
BATCH_SIZE = 512
NUM_WORKERS = 8
EPOCHS = 100
LEARNING_RATE_QUANTUM = 0.001
LEARNING_RATE_CLASSICAL = 0.001
EARLY_STOPPING_PATIENCE = 15
```

---

## 🔧 System Requirements

### Hardware
- **GPU:** RTX 3080 (12GB VRAM) ✓ Available
- **CPU:** Multi-core for parallel data loading ✓ Available
- **CUDA:** Version 13.0 ✓ Available

### Software
- **PennyLane:** v0.43.1 ✓ Installed
- **PyTorch:** CUDA-enabled ✓ Available
- **PennyLane-Lightning-GPU:** ✓ Available and tested

---

## 🚦 Running the Experiment

### Option 1: Jupyter Notebook (Recommended)
```bash
# Open the notebook
jupyter notebook compare_ligand_pocket_quantum_vs_classical.ipynb

# Run all cells in order
# The notebook is pre-configured with all optimizations
```

### Option 2: Parallel Training Script
```bash
# Train both models simultaneously
cd /home/priyanshu/QuantumGNN-/Actuall
python train_parallel.py
```

---

## 📈 What to Measure

### Primary Metrics
- **Validation AUC** (main comparison metric)
- **Validation Accuracy**
- **F1 Score**

### Secondary Metrics
- **Training time per epoch**
- **GPU utilization** (use `nvidia-smi` to monitor)
- **Convergence speed** (epochs to best AUC)

### Quantum-Specific Analysis
- **Quantum advantage:** `(AUC_quantum - AUC_classical) / AUC_classical × 100%`
- **Parameter efficiency:** Performance per trainable parameter
- **Expressivity gain:** Benefit from increased depth and qubits

---

## ✅ Pre-Flight Checklist

- [x] `lightning.gpu` device available and tested
- [x] RTX 3080 GPU detected (12GB VRAM)
- [x] CUDA 13.0 installed
- [x] Notebook updated with 8 qubits, 4 layers
- [x] Model code updated to accept `quantum_device` parameter
- [x] Fallback to `lightning.qubit` implemented
- [x] Batch size optimized to 512
- [x] DataLoader workers set to 8
- [x] cuDNN benchmark mode enabled

---

## 🎯 Success Criteria

**Quantum advantage demonstrated if:**
- AUC_quantum > AUC_classical by >2%
- Quantum model shows better generalization (smaller gap between train/val)
- Results are reproducible (fixed seed = 42069)

**Experiment valid if:**
- Both models train to convergence (or early stopping)
- No NaN losses or gradient explosions
- GPU utilization >30% during training

---

## 📊 Expected Results Location

All results will be saved to:
```
./ligand_pocket_comparison_results/
├── quantum_best.pt          # Best quantum model checkpoint
├── classical_best.pt        # Best classical model checkpoint
├── quantum_history.json     # Training history (quantum)
├── classical_history.json   # Training history (classical)
├── comparison_plots.png     # Visualization of results
└── comparison_results.csv   # Final metrics summary
```

---

## 🔬 Experiment Status

**Status:** ✅ Ready to Run
**Configuration Date:** 2025-12-02
**Estimated Runtime:** ~8-10 minutes per epoch × 100 epochs = ~13-17 hours (with early stopping: likely 5-10 hours)

---

## 🎓 Scientific Notes

This experiment tests the hypothesis that **quantum entanglement in the interaction layer** can capture complex non-linear relationships between ligand molecular features and protein pocket descriptors that classical MLPs cannot efficiently represent.

The `StronglyEntanglingLayers` ansatz with depth 4 creates a highly expressive variational quantum circuit that can potentially model complex quantum correlations in the combined ligand-pocket feature space.

**Good luck! 🚀**
