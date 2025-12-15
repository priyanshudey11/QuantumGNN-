# Getting Started with QuantumGNN

This guide will help you get the QuantumGNN project running on your machine in under 10 minutes.

---

## Prerequisites

Before you begin, make sure you have:
- Python 3.10 or higher
- 8 GB RAM minimum
- 5 GB free disk space (for environment and dependencies)

Optional but recommended:
- NVIDIA GPU with CUDA support (for faster training)
- Apple Silicon Mac (M1/M2/M3/M4) for MPS acceleration

---

## Quick Setup (5 minutes)

### Step 1: Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/QuantumGNN.git
cd QuantumGNN

# Create a fresh Python environment
conda create -n quantum python=3.11 -y
conda activate quantum

# Install PyTorch (choose one based on your system)
# For CPU only:
pip install torch torchvision torchaudio

# For NVIDIA GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For macOS (Apple Silicon):
pip install torch torchvision torchaudio

# Install remaining dependencies
pip install -r requirements.txt
```

### Step 2: Get Sample Data

For a quick test, the code includes a small sample mode that uses only 50 protein structures:

```python
# This is already configured in test.ipynb
processor.load_data(max_samples=50)  # Fast: ~10 seconds, 728 interactions
```

To use the full CDPPILBP dataset (optional, for production use):

```bash
# Download (11.1 GB, may take 10-30 minutes)
wget https://zenodo.org/records/10805580/files/CDPPILBP.tar.gz

# Extract (90 GB uncompressed)
tar -xzf CDPPILBP.tar.gz

# Update data path in test.ipynb
DATA_DIR = "/path/to/extracted/CDPPILBP"
```

---

## Running Your First Experiment (3 minutes)

### Option 1: Use the Jupyter Notebook (Recommended)

```bash
# Start Jupyter
jupyter notebook

# Open test.ipynb in your browser
# Run all cells (Cell → Run All)
```

The notebook will:
1. Auto-detect your hardware (CPU/GPU)
2. Load sample data (50 proteins, 728 interactions)
3. Train both quantum and classical models
4. Generate comparison plots
5. Save results to `ligand_pocket_comparison_results/`

Expected output:
```
HARDWARE DETECTED
Platform:     Darwin / Linux / Windows
CPU:          [Your CPU]
Device:       [cuda / mps / cpu]
Batch Size:   [Auto-optimized]
Workers:      [Auto-optimized]

Training QUANTUM
Parameters: 10,694
Epoch 1/100
  Val AUC: 0.55 → 0.62 → 0.68 (improving)

Training CLASSICAL
Parameters: 11,207
Epoch 1/100
  Val AUC: 0.51 → 0.58 → 0.61 (improving)
```

### Option 2: Python Script

Create `quick_test.py`:

```python
from hardware_optimizer import setup_environment
from data import LigandPocketDataProcessor, LigandPocketDataset, collate_fn
from model import LigandPocketQGNN
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# 1. Setup
hw_info, config = setup_environment()
DATA_DIR = "/path/to/data"  # Update this

# 2. Load data (small sample)
processor = LigandPocketDataProcessor(DATA_DIR, seed=42)
processor.load_data(max_samples=50)
interactions = processor.get_dataset()

# 3. Split
train_ints, temp = train_test_split(interactions, test_size=0.2, random_state=42)
val_ints, test_ints = train_test_split(temp, test_size=0.5, random_state=42)

# 4. Create datasets
train_dataset = LigandPocketDataset(processor, train_ints)
val_dataset = LigandPocketDataset(processor, val_ints)

train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=True,
    collate_fn=collate_fn, num_workers=config['num_workers'],
    pin_memory=config['pin_memory'], persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, batch_size=config['batch_size'], shuffle=False,
    collate_fn=collate_fn, num_workers=config['num_workers'],
    pin_memory=config['pin_memory'], persistent_workers=True
)

# 5. Get input dimensions
sample_ligand = processor.ligands[interactions[0].ligand_id]
sample_pocket = processor.pockets[interactions[0].pocket_id]
ligand_dim = sample_ligand.atom_features.shape[1]  # 10
pocket_dim = sample_pocket.to_vector().shape[0]    # 19

# 6. Create quantum model
model = LigandPocketQGNN(
    ligand_in_dim=ligand_dim,
    pocket_in_dim=pocket_dim,
    hidden_dim=64,
    n_qubits=6,
    n_qlayers=2,
    use_quantum=True,
    quantum_device=config['quantum_device'],
    use_parallel=True
).to(hw_info['device'])

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
print("Ready to train! See test.ipynb for full training loop.")
```

Run it:
```bash
python quick_test.py
```

---

## Understanding the Output

### Training Metrics

During training, you'll see:

```
Epoch 1/100
  Train: Loss=0.693, Acc=0.500
  Val:   Loss=0.686, Acc=0.575, AUC=0.562, F1=0.000
    New best AUC: 0.562

Epoch 5/100
  Train: Loss=0.650, Acc=0.580
  Val:   Loss=0.645, Acc=0.620, AUC=0.680, F1=0.585
    New best AUC: 0.680
```

**What to look for:**
- **AUC (Area Under Curve)**: Primary metric, higher is better (0-1 scale)
- **Accuracy**: Percentage correct predictions
- **F1**: Balance of precision and recall
- Early stopping triggers after 15 epochs without AUC improvement

### Saved Files

After training, check `ligand_pocket_comparison_results/`:

```
quantum_best.pt          # Best quantum model weights
quantum_history.json     # Training curves (loss, AUC, etc.)
classical_best.pt        # Best classical model weights
classical_history.json   # Classical training curves
comparison_plots.png     # Side-by-side performance graphs
```

---

## Common Issues and Quick Fixes

### Issue 1: "No module named 'pennylane'"

**Fix:**
```bash
pip install pennylane pennylane-lightning
```

### Issue 2: "CUDA out of memory"

**Fix:** Reduce batch size in notebook:
```python
# In test.ipynb, Cell 3
config['batch_size'] = 256  # or lower
```

### Issue 3: "MPS backend assertion error"

This is already handled automatically. The code falls back to CPU for sparse operations.

### Issue 4: Training is very slow

**Causes and fixes:**
- **Too many workers**: Reduce `num_workers` to 4
- **Large dataset**: Use `max_samples=50` for testing
- **CPU-only quantum**: Expected - quantum simulation is slow on CPU. Use classical model for speed testing.

### Issue 5: "FileNotFoundError: descriptor CSV not found"

**Fix:** Check your data directory structure:
```
DATA_DIR/
└── [protein_name]/
    └── results/
        └── [pocket_id]/
            ├── [pocket_id]_descriptors_3d.csv  ← Must exist
            └── *.mol2                          ← Ligand files
```

---

## Next Steps

Once the basic example works:

1. **Try the classical model**: Set `use_quantum=False` to compare
2. **Load more data**: Increase `max_samples` (100, 500, 1000, etc.)
3. **Adjust hyperparameters**: Change `hidden_dim`, `n_qubits`, `n_qlayers`
4. **Read the docs**:
   - [WIKI.md](WIKI.md) - Full technical documentation
   - [README.md](README.md) - Project overview
   - [PAPER_CODE_ALIGNMENT.md](PAPER_CODE_ALIGNMENT.md) - Research paper analysis

---

## Quick Reference: Key Parameters

### Data Loading
```python
processor.load_data(max_samples=50)  # 50 = fast test, None = full dataset
```

### Model Architecture
```python
LigandPocketQGNN(
    ligand_in_dim=10,      # Fixed: atom type encoding
    pocket_in_dim=19,      # Fixed: pocket features
    hidden_dim=64,         # Increase for more capacity (32, 128, 256)
    n_qubits=6,            # Quantum circuit width (4-8 recommended)
    n_qlayers=2,           # Circuit depth (1-4 recommended)
    use_quantum=True,      # False = classical MLP
    use_parallel=True      # True = multi-threaded quantum (faster)
)
```

### Training
```python
EPOCHS = 100                    # Max training epochs
LEARNING_RATE = 0.001          # Adam learning rate
EARLY_STOPPING_PATIENCE = 15   # Stop after N epochs without improvement
```

### Hardware
```python
# Auto-detected, but can override:
config['batch_size'] = 512     # Larger = faster if you have RAM/VRAM
config['num_workers'] = 6      # More = faster data loading (up to CPU cores)
```

---

## Expected Performance

On a typical system (Apple M4, 24 GB RAM, batch_size=1024):

| Model | Samples | Epochs | Time/Epoch | Final Val AUC |
|-------|---------|--------|------------|---------------|
| Quantum | 728 (50 proteins) | ~15-25 | 3-5 sec | 0.65-0.70 |
| Classical | 728 (50 proteins) | ~25-35 | 0.5-1 sec | 0.60-0.65 |

Full dataset (244k interactions):
- Training time: 5-20 minutes per epoch (depending on hardware)
- Expected to converge in 30-100 epochs
- Final AUC: 0.70-0.85 (varies by hyperparameters)

---

## Getting Help

If you're stuck:

1. **Check the error message**: Most issues are self-explanatory
2. **Verify data structure**: Use `processor.load_data(max_samples=1)` to test
3. **Try CPU mode**: Set `DEVICE = 'cpu'` to rule out GPU issues
4. **Reduce complexity**: Use `max_samples=10`, `batch_size=32` for debugging
5. **Read the WIKI**: [WIKI.md](WIKI.md) has detailed troubleshooting

Still stuck? Open an issue on GitHub with:
- Your system info (OS, Python version, GPU)
- Full error message
- Minimal code to reproduce the issue

---

## Summary: Fastest Path to Results

```bash
# 1. Setup (2 minutes)
conda create -n quantum python=3.11 -y
conda activate quantum
pip install torch pennylane scikit-learn pandas numpy tqdm matplotlib

# 2. Run (30 seconds)
jupyter notebook test.ipynb
# Click "Run All"

# 3. Wait (3-5 minutes)
# Training completes, plots generated

# 4. Check results
open ligand_pocket_comparison_results/comparison_plots.png
```

That's it. You now have a working quantum-classical GNN comparison.

---

**Last Updated**: December 14, 2025
**Version**: 1.0.0
