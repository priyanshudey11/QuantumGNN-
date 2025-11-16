# Jupyter Notebooks Guide

This project includes two Jupyter notebooks for different purposes.

## 📓 Available Notebooks

### 1. `train_model.ipynb` ⭐ **START HERE FOR TRAINING**

**Purpose**: Streamlined training workflow with easy configuration

**Best for**:
- Training your model quickly
- Running experiments with different hyperparameters
- Production training runs
- Saving and evaluating trained models

**Features**:
- ✅ **Easy configuration** at the top of the notebook
- ✅ **Automated workflow**: Just run all cells
- ✅ **Model saving**: Automatically saves model, checkpoint, and results
- ✅ **Performance metrics**: Complete evaluation with visualizations
- ✅ **Experiment tracking**: Timestamped experiment names
- ✅ **Clean output**: Focused on training and results

**Configuration (First Cell)**:
```python
# Modify these parameters:
MAX_DRUGS = 100              # Number of drugs to load
N_PATIENTS = 200             # Number of patients
NUM_QUBITS = 4               # Qubits per side
NUM_QLAYERS = 2              # Variational layers
EPOCHS = 100                 # Training epochs
USE_QUANTUM = True           # Quantum vs Classical
```

**Usage**:
```bash
jupyter notebook train_model.ipynb
# → Modify configuration in first cell
# → Run All Cells
# → Models saved to ./saved_models/
```

---

### 2. `tutorial.ipynb` **FOR LEARNING**

**Purpose**: Comprehensive tutorial with explanations

**Best for**:
- Learning how the system works
- Understanding each component
- Exploring the data and model
- Comparing quantum vs classical
- Educational purposes

**Features**:
- ✅ **Step-by-step explanations**
- ✅ **Data exploration** with visualizations
- ✅ **Model architecture** explanation
- ✅ **Quantum vs Classical comparison**
- ✅ **Interactive predictions**
- ✅ **Multiple examples** and use cases

**Usage**:
```bash
jupyter notebook tutorial.ipynb
# → Read explanations in each section
# → Run cells interactively
# → Experiment with code
```

---

## 🎯 Quick Decision Guide

### I want to...

**Train a model quickly**
→ Use `train_model.ipynb`
```bash
jupyter notebook train_model.ipynb
```

**Learn how everything works**
→ Use `tutorial.ipynb`
```bash
jupyter notebook tutorial.ipynb
```

**Run multiple experiments**
→ Use `train_model.ipynb` with different configs
```python
# Experiment 1: Classical baseline
USE_QUANTUM = False
EXPERIMENT_NAME = "classical_baseline"

# Experiment 2: Quantum with 4 qubits
USE_QUANTUM = True
NUM_QUBITS = 4

# Experiment 3: Quantum with 6 qubits
USE_QUANTUM = True
NUM_QUBITS = 6
```

**Compare quantum vs classical**
→ Use `tutorial.ipynb` (has built-in comparison)

**Production training**
→ Use `train_model.ipynb` or command line:
```bash
python drug_patient_qgnn/examples/run_pipeline.py
```

---

## 📊 Workflow Examples

### Example 1: Quick Training Run

```bash
# 1. Open training notebook
jupyter notebook train_model.ipynb

# 2. Modify configuration (first cell):
EPOCHS = 50
NUM_QUBITS = 4
USE_QUANTUM = True

# 3. Run all cells (Cell → Run All)

# 4. Find results in ./saved_models/
```

### Example 2: Learning Workflow

```bash
# 1. Start with tutorial
jupyter notebook tutorial.ipynb

# 2. Run through all sections to understand:
#    - Data loading
#    - Graph construction
#    - Model architecture
#    - Training process
#    - Evaluation

# 3. Then move to training notebook for experiments
jupyter notebook train_model.ipynb
```

### Example 3: Hyperparameter Search

```python
# In train_model.ipynb, run multiple times with different configs:

# Run 1:
NUM_QUBITS = 2
NUM_QLAYERS = 1
# → Run all cells → saved as quantum_q2_l1_20231115_123456

# Run 2:
NUM_QUBITS = 4
NUM_QLAYERS = 2
# → Run all cells → saved as quantum_q4_l2_20231115_123512

# Run 3:
NUM_QUBITS = 6
NUM_QLAYERS = 3
# → Run all cells → saved as quantum_q6_l3_20231115_123598

# Compare results in ./saved_models/
```

---

## 📂 Output Structure

When you run `train_model.ipynb`, it creates:

```
saved_models/
├── quantum_q4_l2_20231115_143022_model.pt        # Model weights
├── quantum_q4_l2_20231115_143022_checkpoint.pt   # Full checkpoint
├── quantum_q4_l2_20231115_143022_graph.pkl       # Data graph
├── quantum_q4_l2_20231115_143022_history.json    # Training history
├── quantum_q4_l2_20231115_143022_config.json     # Configuration
└── quantum_q4_l2_20231115_143022_evaluation.png  # Plots
```

---

## 🔧 Customization

### Modify Training Configuration

Edit the first cell of `train_model.ipynb`:

```python
# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Data
DATA_DIR = "/media/priyanshu/SD/othercode/data"
MAX_DRUGS = 100              # ← Change this
N_PATIENTS = 200             # ← Change this
INTERACTION_RATE = 0.05

# Model
NUM_QUBITS = 4               # ← Change this (2, 4, 6, 8)
NUM_QLAYERS = 2              # ← Change this (1, 2, 3)
USE_QUANTUM = True           # ← Change this (True/False)

# Training
EPOCHS = 100                 # ← Change this
BATCH_SIZE = 32              # ← Change this
LEARNING_RATE = 0.001        # ← Change this
VAL_SPLIT = 0.2

# Other
SEED = 42                    # For reproducibility
DEVICE = None                # Auto-detect (or 'cuda', 'cpu', 'mps')
```

### Add Custom Analysis

Add new cells at the end of either notebook:

```python
# Example: Analyze feature importance
drug_features.requires_grad = True
output = model(drug_features, patient_features)
output.sum().backward()
importance = drug_features.grad.abs().mean(dim=0)

import matplotlib.pyplot as plt
plt.bar(range(len(importance)), importance.detach())
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Drug Feature Importance')
plt.show()
```

---

## 💡 Tips

### For Training Notebook:

1. **Start small**: Use small epochs (10-20) for testing
2. **Increase gradually**: Once working, increase to 100+ epochs
3. **Monitor GPU**: Check device info cell to verify GPU usage
4. **Save configurations**: The config.json helps track experiments
5. **Compare experiments**: Load history.json files to compare runs

### For Tutorial Notebook:

1. **Read all explanations**: Don't skip the markdown cells
2. **Run interactively**: Execute cells one by one
3. **Experiment**: Modify code in cells to see effects
4. **Use for reference**: Keep it open while developing

---

## 🐛 Troubleshooting

### Notebook won't start
```bash
# Install Jupyter
pip install jupyter notebook

# Start
jupyter notebook train_model.ipynb
```

### Kernel dies during training
- Reduce `BATCH_SIZE` (try 16 or 8)
- Reduce `NUM_QUBITS` (try 2 or 3)
- Use `USE_QUANTUM = False` for classical mode
- Set `DEVICE = 'cpu'` to avoid GPU memory issues

### "Module not found" error
```bash
# Make sure package is installed
cd /home/priyanshu/QuantumGNN-/Actuall
pip install -e .
```

### Slow quantum training
- Reduce `NUM_QUBITS` (each qubit doubles computation)
- Reduce `NUM_QLAYERS`
- Use smaller `BATCH_SIZE`
- Try `USE_QUANTUM = False` for baseline

---

## 📊 Expected Training Times

With default configuration (4 qubits, 2 layers, 100 epochs):

| Mode | Hardware | Time |
|------|----------|------|
| Classical | CPU | ~5-10 min |
| Classical | GPU | ~2-5 min |
| Quantum | CPU | ~20-40 min |
| Quantum | GPU | ~10-20 min |

For 6+ qubits, quantum mode can take 1-2 hours.

---

## 🎓 Learning Path

**Recommended order**:

1. **Day 1**: Run `tutorial.ipynb` completely
   - Understand data structure
   - See how model works
   - Review visualizations

2. **Day 2**: Run `train_model.ipynb` with defaults
   - See full training workflow
   - Check saved outputs
   - Review metrics

3. **Day 3+**: Experiment with `train_model.ipynb`
   - Try different qubits (2, 4, 6)
   - Compare quantum vs classical
   - Tune hyperparameters

---

## 📞 Need Help?

- **For training issues**: Check `train_model.ipynb` cell outputs
- **For understanding**: Review `tutorial.ipynb` explanations
- **For code details**: See `IMPLEMENTATION_SUMMARY.md`
- **For dataset**: See `DATASET_NOTES.md`

---

**Happy Training!** 🚀⚛️
