# QuantumGNN Setup Guide

Complete installation guide for the Quantum Graph Neural Network project.

## System Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 3080)
- **CUDA**: Version 12.x or compatible
- **Python**: 3.11 (recommended)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for dependencies and data

## Quick Setup (If you already have the environment)

```bash
# Activate environment
conda activate quantum

# Verify GPU setup
python test_gpu_setup.py

# Start training
jupyter notebook compare_quantum_vs_classical_OPTIMIZED.ipynb
```

## Fresh Installation

### Step 1: Create Conda Environment

```bash
# Create new environment with Python 3.11
conda create -n quantum python=3.11 -y

# Activate environment
conda activate quantum
```

### Step 2: Install PyTorch with CUDA

```bash
# For CUDA 12.1 (adjust based on your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output: `CUDA available: True`

### Step 3: Install PennyLane with GPU Support

```bash
# Install PennyLane and GPU-accelerated backend
pip install pennylane pennylane-lightning pennylane-lightning-gpu

# Verify PennyLane GPU
python -c "import pennylane as qml; dev = qml.device('lightning.gpu', wires=2); print('PennyLane GPU OK')"
```

Expected output: `PennyLane GPU OK`

### Step 4: Install Remaining Dependencies

```bash
# Install from requirements file
pip install numpy pandas scikit-learn matplotlib seaborn tqdm jupyter jupyterlab ipykernel ipywidgets
```

Or use the requirements file:
```bash
pip install -r requirements-minimal.txt
```

### Step 5: Verify Complete Setup

```bash
# Run comprehensive GPU test
python test_gpu_setup.py
```

You should see all tests passing with ✅ marks.

## Project Structure

```
Actuall/
├── drug_patient_qgnn/          # Core package
│   ├── __init__.py
│   ├── model.py                # Quantum GNN model
│   ├── data_processing.py      # Data loading
│   └── utils.py                # Helper functions
│
├── compare_quantum_vs_classical_OPTIMIZED.ipynb  # Main training notebook
├── test_gpu_setup.py           # GPU verification script
├── requirements.txt            # Detailed requirements with versions
├── requirements-minimal.txt    # Minimal requirements
└── GPU_TRAINING_FIXED.md       # Training fix documentation
```

## Running Training

### Option 1: Jupyter Notebook (Recommended)

```bash
# Start Jupyter
jupyter notebook

# Open: compare_quantum_vs_classical_OPTIMIZED.ipynb
# Run all cells (Kernel → Restart & Run All)
```

### Option 2: Python Script

If you prefer running from command line:
```bash
jupyter nbconvert --to script compare_quantum_vs_classical_OPTIMIZED.ipynb
python compare_quantum_vs_classical_OPTIMIZED.py
```

## Configuration

Key settings in the notebook (Cell 1):

```python
# Data
MAX_DRUGS = 1000              # Dataset size
BATCH_SIZE = 128              # Batch size

# Model
NUM_QUBITS = 6                # Qubits per side
NUM_QLAYERS = 2               # Quantum layers

# Hardware
DEVICE = 'cuda'               # PyTorch device
QUANTUM_DEVICE = 'lightning.gpu'  # PennyLane device
NUM_WORKERS = 0               # IMPORTANT: Keep at 0 for quantum!
```

**CRITICAL**: `NUM_WORKERS` must be 0 to avoid multiprocessing issues with quantum circuits.

## Expected Training Time

- **Quantum Model**: ~8-10 hours
- **Classical Model**: ~6-8 hours
- **Total**: ~14-18 hours for full comparison

Progress is auto-saved every epoch to `./optimized_comparison_results/`

## Troubleshooting

### Issue: DataLoader workers crash
**Symptom**: `RuntimeError: DataLoader worker exited unexpectedly`
**Solution**: Set `NUM_WORKERS = 0` in configuration

### Issue: CUDA out of memory
**Symptom**: `RuntimeError: CUDA out of memory`
**Solution**: Reduce `BATCH_SIZE` (try 64 or 32)

### Issue: PennyLane GPU not available
**Symptom**: `Device lightning.gpu not found`
**Solution**:
1. Check CUDA is installed: `nvidia-smi`
2. Reinstall: `pip install --upgrade pennylane-lightning-gpu`
3. Fallback to CPU: `QUANTUM_DEVICE = 'lightning.qubit'`

### Issue: Slow training
**Symptom**: Each batch takes >30 seconds
**Solution**:
- Verify GPU is being used: `nvidia-smi` (should show GPU utilization)
- Check you're using `lightning.gpu` not `default.qubit`
- Reduce `NUM_QLAYERS` or `NUM_QUBITS`

## Monitoring Training

### Real-time Progress
The notebook shows:
- Batch-level progress bars (tqdm)
- Epoch-level metrics (loss, accuracy, AUC)
- Estimated time remaining

### Check GPU Usage
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### View Saved Results
```bash
# Check auto-saved history
cat ./optimized_comparison_results/quantum_history.json

# View latest metrics
tail -f ./optimized_comparison_results/quantum_history.json
```

## Output Files

Training generates:
- `quantum_best.pt` - Best quantum model checkpoint
- `classical_best.pt` - Best classical model checkpoint
- `quantum_history.json` - Training metrics (auto-saved every epoch)
- `classical_history.json` - Training metrics
- `final_results.json` - Complete comparison results
- `quantum_vs_classical_comparison.png` - Results plot

## Environment Variables

Optional optimization flags (already set in notebook):
```bash
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
```

## Version Information

Tested configuration:
- **Python**: 3.11.x
- **PyTorch**: 2.9.1
- **PennyLane**: 0.43.1
- **CUDA**: 12.1
- **GPU**: NVIDIA RTX 3080

## Getting Help

1. **GPU Setup Issues**: Run `python test_gpu_setup.py` and check output
2. **Training Issues**: See [GPU_TRAINING_FIXED.md](GPU_TRAINING_FIXED.md)
3. **DataLoader Errors**: Verify `NUM_WORKERS = 0`

## Additional Resources

- [PennyLane Documentation](https://pennylane.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads)
