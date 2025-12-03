# 🚀 Quick Start Guide

Get your quantum GNN training running in 5 minutes.

## ✅ Prerequisites

- NVIDIA GPU with CUDA 12.x
- Conda installed
- ~16GB RAM

## 📦 Installation (Already Done)

Your environment is already set up! Skip to [Running Training](#running-training).

<details>
<summary>🔧 Fresh Installation (click to expand)</summary>

```bash
# Create environment
conda create -n quantum python=3.11 -y
conda activate quantum

# Install PyTorch + CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install PennyLane + GPU
pip install pennylane pennylane-lightning pennylane-lightning-gpu

# Install dependencies
pip install -r requirements-minimal.txt
```

</details>

## 🧪 Verify Setup

**IMPORTANT**: Run this before training to ensure everything works:

```bash
conda activate quantum
python test_gpu_setup.py
```

Expected output: All tests should show ✅

## 🎯 Running Training

### Option 1: Jupyter Notebook (Recommended)

```bash
conda activate quantum
jupyter notebook
```

Then open: `compare_quantum_vs_classical_OPTIMIZED.ipynb`

Click: **Kernel → Restart & Run All**

### Option 2: Command Line

```bash
conda activate quantum
jupyter nbconvert --to script compare_quantum_vs_classical_OPTIMIZED.ipynb
python compare_quantum_vs_classical_OPTIMIZED.py
```

## ⏱️ Expected Runtime

- **Quantum Model**: ~8-10 hours
- **Classical Model**: ~6-8 hours
- **Total**: ~14-18 hours

Progress auto-saves every epoch to `./optimized_comparison_results/`

## 📊 Monitor Progress

### In Jupyter
- Real-time progress bars
- Epoch metrics
- Best AUC tracking

### GPU Usage
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Check Saved Results
```bash
# View training history
cat ./optimized_comparison_results/quantum_history.json
```

## 🐛 Troubleshooting

### ❌ DataLoader workers crash
Already fixed! `NUM_WORKERS = 0` is set in the notebook.

### ❌ CUDA out of memory
Edit notebook, reduce `BATCH_SIZE`:
```python
BATCH_SIZE = 64  # or 32
```

### ❌ Tests fail
See [GPU_TRAINING_FIXED.md](GPU_TRAINING_FIXED.md)

## 📁 Output Files

Results saved to `./optimized_comparison_results/`:

- `quantum_best.pt` - Best model checkpoint
- `quantum_history.json` - Training metrics
- `final_results.json` - Final comparison
- `quantum_vs_classical_comparison.png` - Results plot

## 📚 More Documentation

- [SETUP.md](SETUP.md) - Detailed installation
- [GPU_TRAINING_FIXED.md](GPU_TRAINING_FIXED.md) - Troubleshooting
- [requirements.txt](requirements.txt) - Package versions

## 🎉 That's It!

You're ready to train. Good luck with your quantum GNN! 🚀
