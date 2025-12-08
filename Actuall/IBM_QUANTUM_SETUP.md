# IBM Quantum Setup Guide for Ligand-Pocket QGNN

This guide explains how to set up and run your Ligand-Pocket QGNN model on IBM Quantum hardware.

## Prerequisites

1. **IBM Quantum Account**: Sign up at https://quantum.ibm.com/
2. **Required Python Packages**:
```bash
pip install qiskit qiskit-ibm-runtime pennylane pennylane-qiskit
```

## Setup Steps

### 1. Get Your IBM Quantum API Token

1. Go to https://quantum.ibm.com/
2. Log in or create an account
3. Navigate to your account settings
4. Copy your API token

### 2. Configure IBM Quantum Access

Open a Python shell or Jupyter notebook and run:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save your IBM Quantum credentials (only need to do this once)
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_IBM_QUANTUM_TOKEN_HERE",
    overwrite=True
)
```

### 3. Verify Connection

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
print("Available backends:")
for backend in service.backends():
    print(f"  {backend.name} - {backend.num_qubits} qubits")
```

## Running the Notebook

### File Structure

```
Actuall/
├── compare_ligand_pocket_quantum_vs_classical_IBM.ipynb  # IBM Quantum notebook
├── ligand_pocket_qgnn/
│   ├── model_ibm.py          # IBM Quantum model
│   ├── model.py              # Original model (for classical)
│   └── data.py               # Data processing
└── ligand_pocket_comparison_results_IBM/  # Results directory (created automatically)
```

### Configuration

In the notebook, modify cell 6 to select your backend:

```python
# IBM Quantum Backend Selection
IBM_BACKEND = 'ibmq_qasm_simulator'  # Options:

# Simulators (FREE, fast):
# - 'ibmq_qasm_simulator'      : IBM's cloud QASM simulator
# - 'simulator_statevector'    : Statevector simulator

# Real Quantum Hardware (requires queue time):
# - 'ibm_brisbane'            : 127-qubit quantum computer
# - 'ibm_kyoto'               : 127-qubit quantum computer
# - 'ibm_osaka'               : 127-qubit quantum computer
```

### Training Configuration

**For Simulators** (Recommended for testing):
```python
IBM_BACKEND = 'ibmq_qasm_simulator'
N_QUBITS = 6
N_QLAYERS = 2
BATCH_SIZE = 128  # Auto-configured based on backend
```

**For Real Quantum Hardware**:
```python
IBM_BACKEND = 'ibm_brisbane'  # Or other real hardware
N_QUBITS = 6
N_QLAYERS = 2
BATCH_SIZE = 16  # Smaller for real hardware
```

## Important Differences from Simulated Version

| Feature | Simulator Version | IBM Quantum Version |
|---------|------------------|-------------------|
| Backend | `lightning.gpu` | IBM Quantum backends |
| Qubits | 8 | 6 (reduced for compatibility) |
| Circuit Depth | 4 layers | 2 layers (reduced for speed) |
| Batch Size | 8192 | 128 (simulator) / 16 (hardware) |
| Speed | Very fast | Slow (hardware has queue times) |
| Shots | N/A (statevector) | 8192 measurements |
| Differentiation | Automatic | Parameter-shift rule |

## Expected Training Times

### Simulators:
- `ibmq_qasm_simulator`: ~5-10 minutes per epoch
- Batch size: 128
- Total training: ~8-16 hours

### Real Quantum Hardware:
- Queue wait time: Varies (minutes to hours)
- Execution time: ~30-120 minutes per epoch
- Total training: Days to weeks
- **Recommendation**: Start with 1-2 epochs for testing

## Running the Training

1. **Start Jupyter**:
```bash
cd /home/priyanshu/QuantumGNN-/Actuall
jupyter notebook compare_ligand_pocket_quantum_vs_classical_IBM.ipynb
```

2. **Run cells in order**:
   - Cell 1: Imports
   - Cell 2: IBM Quantum setup (verify connection)
   - Cell 3: Configuration
   - Cell 4-13: Data loading and training functions
   - Cell 14: **Train IBM Quantum model** (this is the main training)
   - Cell 16: Train classical model for comparison
   - Cell 18-22: Results and visualization

## Monitoring Progress

The notebook will show:
- Connection status to IBM backend
- Queue position (for real hardware)
- Training progress per batch
- Validation metrics after each epoch
- Best model checkpoints saved automatically

## Troubleshooting

### Error: "No IBM Quantum account found"
```python
# Re-run the setup with your token
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_TOKEN",
    overwrite=True
)
```

### Error: "Backend not found"
```python
# List available backends
service = QiskitRuntimeService()
for backend in service.backends():
    print(backend.name)
```

### Training is too slow
- Switch to `ibmq_qasm_simulator` for faster testing
- Reduce batch size
- Reduce number of epochs
- Consider training on a subset of data first

### Out of memory
- Reduce `BATCH_SIZE` in cell 8
- Reduce `N_QUBITS` (requires model architecture change)

## Resuming Training

If training is interrupted, you can resume from checkpoint:

In cell 14, change:
```python
quantum_history, quantum_best_auc = train_model(
    quantum_model,
    train_loader,
    val_loader,
    LEARNING_RATE_QUANTUM,
    "quantum_ibm",
    DEVICE,
    resume_from_checkpoint=True  # Set to True
)
```

## Comparison with Original Notebook

| Feature | Original | IBM Quantum |
|---------|----------|-------------|
| File | `compare_ligand_pocket_quantum_vs_classical.ipynb` | `compare_ligand_pocket_quantum_vs_classical_IBM.ipynb` |
| Model | `LigandPocketQGNN` | `LigandPocketQGNN_IBM` |
| Backend | `lightning.gpu` | IBM Quantum |
| Results Dir | `ligand_pocket_comparison_results/` | `ligand_pocket_comparison_results_IBM/` |

## Cost Considerations

- **IBM Quantum Simulators**: FREE, unlimited access
- **IBM Quantum Hardware**:
  - Free tier: 10 minutes/month of quantum compute time
  - Premium access: Contact IBM for pricing
  - Academic access: Often free with university partnership

## Best Practices

1. **Start with simulators** to verify your code works
2. **Test with small batch sizes** before full training
3. **Use sessions** for better performance (enabled by default)
4. **Monitor your usage** on the IBM Quantum dashboard
5. **Save checkpoints frequently** in case of interruptions
6. **Test with 1-2 epochs** on real hardware before full training

## Results

Results will be saved to:
- `ligand_pocket_comparison_results_IBM/quantum_ibm_best.pt` - Best model checkpoint
- `ligand_pocket_comparison_results_IBM/quantum_ibm_history.json` - Training history
- `ligand_pocket_comparison_results_IBM/comparison_plots_{backend}.png` - Plots
- `ligand_pocket_comparison_results_IBM/comparison_results_{backend}.csv` - Metrics
- `ligand_pocket_comparison_results_IBM/experiment_metadata_{backend}.json` - Metadata

## Support

- IBM Quantum Documentation: https://docs.quantum.ibm.com/
- PennyLane-Qiskit: https://docs.pennylane.ai/projects/qiskit/
- Qiskit Runtime: https://qiskit.org/ecosystem/ibm-runtime/

## Quick Start Command

```bash
# Full setup and run
cd /home/priyanshu/QuantumGNN-/Actuall
jupyter notebook compare_ligand_pocket_quantum_vs_classical_IBM.ipynb

# In the notebook:
# 1. Run cell 2 to verify IBM Quantum connection
# 2. Run all cells in order to start training
```
