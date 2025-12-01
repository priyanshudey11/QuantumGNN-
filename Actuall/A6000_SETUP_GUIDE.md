# A6000 Server Setup Guide

## Quick Start

This guide helps you run the quantum vs classical comparison on your remote A6000 server.

## Option 1: Run as Jupyter Notebook (Recommended for Monitoring)

### 1. Transfer files to A6000 server
```bash
# From your local machine
scp compare_quantum_vs_classical_A6000.ipynb user@a6000-server:/path/to/project/
scp -r drug_patient_qgnn/ user@a6000-server:/path/to/project/
```

### 2. SSH with port forwarding
```bash
ssh -L 8888:localhost:8888 user@a6000-server
```

### 3. Start Jupyter on server
```bash
cd /path/to/project
conda activate quantum_env  # Or your environment name
jupyter lab --no-browser --port=8888
```

### 4. Open notebook in local browser
- Open the URL shown (http://localhost:8888/lab?token=...)
- Navigate to `compare_quantum_vs_classical_A6000.ipynb`
- **IMPORTANT**: Update `DATA_DIR` path in cell 2 to match your server's data location
- Run all cells

---

## Option 2: Run as Python Script (Background Training)

### 1. Convert and transfer
```bash
# Use the provided train_a6000.py script
scp train_a6000.py user@a6000-server:/path/to/project/
scp -r drug_patient_qgnn/ user@a6000-server:/path/to/project/
```

### 2. Run in background with tmux/screen
```bash
ssh user@a6000-server
cd /path/to/project

# Update DATA_DIR in train_a6000.py first!
nano train_a6000.py  # Edit line 22

# Start tmux session
tmux new -s quantum_training

# Run training
conda activate quantum_env
python train_a6000.py

# Detach: Ctrl+B then D
# Reattach: tmux attach -t quantum_training
```

### 3. Monitor progress remotely
```bash
# In another SSH session
watch -n 30 "tail -50 a6000_training.log"

# Or check JSON history
python -c "import json; h=json.load(open('a6000_results/quantum_history.json')); print(f'Epoch {len(h[\"val_auc\"])}: AUC={h[\"val_auc\"][-1]:.4f}')"
```

---

## Important Configuration Changes

### Update DATA_DIR (Required!)
The notebook/script has a placeholder path. You MUST update this:

```python
# Change this line to match your server's data location
DATA_DIR = "/your/actual/path/to/data/drug_data_smiles_no_duplicates"
```

### Verify GPU Availability
```bash
nvidia-smi  # Should show 2× A6000 GPUs
```

### Check Dependencies
```bash
conda activate quantum_env
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import pennylane as qml; print(f'PennyLane: {qml.__version__}')"
```

---

## Expected Performance

- **Batch Size**: 2048 (utilizing 48GB VRAM)
- **Time per Epoch**:
  - Quantum: ~15-20 min (CPU simulation bottleneck)
  - Classical: ~30 sec (multi-GPU accelerated)
- **Total Time**: 2-3 hours with early stopping
- **Storage**: ~200 MB for all results

---

## Monitoring Training

### Check current progress
```bash
ls -lh a6000_results/
cat a6000_training.log | tail -20
```

### Use monitoring script
```bash
# Transfer monitor script
scp early_stopping_monitor.py user@a6000-server:/path/to/project/

# Run in separate terminal
python early_stopping_monitor.py
```

---

## Troubleshooting

### CUDA Out of Memory
Reduce `BATCH_SIZE` from 2048 to 1024 in the script/notebook.

### Module Not Found
```bash
conda activate quantum_env
pip install torch torchvision pennylane rdkit scikit-learn matplotlib
```

### Slow Quantum Training
This is expected! Quantum circuit simulation is CPU-bound (~15-20 min/epoch).
The A6000 GPUs are mainly used for classical model training.

---

## After Training Completes

Results will be saved to `a6000_results/`:
- `quantum_best.pt` - Best quantum model
- `classical_best.pt` - Best classical model
- `quantum_history.json` - Training history
- `classical_history.json` - Training history
- `comparison_plots.png` - Visualization
- `final_results.txt` - Summary statistics

Transfer back to local machine:
```bash
scp -r user@a6000-server:/path/to/project/a6000_results/ ./
```
