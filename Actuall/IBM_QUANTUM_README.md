# IBM Quantum Version - Ligand-Pocket QGNN

This directory contains an IBM Quantum Hardware compatible version of your Ligand-Pocket QGNN comparison notebook.

## What Was Created

### 1. **New Model File**: [model_ibm.py](ligand_pocket_qgnn/model_ibm.py)
   - `LigandPocketQGNN_IBM`: Model class that uses IBM Quantum backends
   - `IBMQuantumInteractionLayer`: Quantum layer configured for IBM hardware
   - Uses `qiskit.remote` device with PennyLane
   - Supports both simulators and real quantum computers

### 2. **New Notebook**: [compare_ligand_pocket_quantum_vs_classical_IBM.ipynb](compare_ligand_pocket_quantum_vs_classical_IBM.ipynb)
   - Modified version of your original comparison notebook
   - Includes IBM Quantum setup and configuration
   - Adjusted batch sizes and parameters for IBM Quantum
   - Saves results to separate directory

### 3. **Setup Guide**: [IBM_QUANTUM_SETUP.md](IBM_QUANTUM_SETUP.md)
   - Complete setup instructions
   - Backend selection guide
   - Troubleshooting tips
   - Best practices

### 4. **Requirements**: [requirements_ibm.txt](requirements_ibm.txt)
   - All IBM Quantum dependencies
   - Install with: `pip install -r requirements_ibm.txt`

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_ibm.txt
```

### 2. Setup IBM Quantum Account
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_IBM_QUANTUM_TOKEN",
    overwrite=True
)
```

Get your token from: https://quantum.ibm.com/

### 3. Run the Notebook
```bash
jupyter notebook compare_ligand_pocket_quantum_vs_classical_IBM.ipynb
```

## Key Differences from Original

| Feature | Original Notebook | IBM Quantum Notebook |
|---------|------------------|---------------------|
| **Quantum Backend** | `lightning.gpu` (local simulator) | IBM Quantum (cloud) |
| **Model Class** | `LigandPocketQGNN` | `LigandPocketQGNN_IBM` |
| **Qubits** | 8 | 6 |
| **Circuit Depth** | 4 layers | 2 layers |
| **Batch Size** | 8192 | 128 (simulator) / 16 (hardware) |
| **Training Speed** | Very fast (~0.3 min/epoch) | Slow (~5-120 min/epoch) |
| **Shots** | N/A (statevector) | 8192 measurements |
| **Differentiation** | Auto | Parameter-shift |
| **Results Directory** | `ligand_pocket_comparison_results/` | `ligand_pocket_comparison_results_IBM/` |

## Available IBM Quantum Backends

### Simulators (FREE, Recommended for Testing)
- **`ibmq_qasm_simulator`**: IBM's cloud QASM simulator (unlimited, fast)
- **`simulator_statevector`**: Statevector simulator

### Real Quantum Computers (Queue Times Apply)
- **`ibm_brisbane`**: 127-qubit quantum computer
- **`ibm_kyoto`**: 127-qubit quantum computer
- **`ibm_osaka`**: 127-qubit quantum computer

Check current availability at: https://quantum.ibm.com/services/resources

## Configuration Example

In the notebook (cell 6):

```python
# For simulator (recommended first):
IBM_BACKEND = 'ibmq_qasm_simulator'
N_QUBITS = 6
N_QLAYERS = 2
BATCH_SIZE = 128  # Auto-configured

# For real quantum hardware:
IBM_BACKEND = 'ibm_brisbane'
N_QUBITS = 6
N_QLAYERS = 2
BATCH_SIZE = 16  # Auto-configured
```

## Expected Performance

### Simulator Training:
- Time per epoch: ~5-10 minutes
- Total training: ~8-16 hours (100 epochs with early stopping)
- Cost: FREE

### Real Quantum Hardware Training:
- Queue time: Varies (minutes to hours)
- Time per epoch: ~30-120 minutes
- Total training: Days to weeks
- Cost: Free tier (10 min/month) or premium access

## Why Use IBM Quantum?

### Advantages:
1. **Real quantum hardware**: Test on actual quantum computers
2. **Quantum noise effects**: See how noise affects your model
3. **Research credibility**: Results from real quantum devices
4. **IBM infrastructure**: Professional-grade quantum systems
5. **Publication ready**: Can cite use of IBM Quantum hardware

### Disadvantages:
1. **Much slower**: Queue times + execution time
2. **Limited free tier**: 10 minutes/month on real hardware
3. **Reduced parameters**: Fewer qubits and layers
4. **Noise**: Real hardware has decoherence and gate errors

## Recommended Workflow

1. **Develop on simulator**: Use your original notebook with `lightning.gpu`
2. **Test on IBM simulator**: Use `ibmq_qasm_simulator` to verify IBM compatibility
3. **Small test on real hardware**: Run 1-2 epochs on real quantum computer
4. **Full training decision**: Based on results, decide if worth the wait/cost

## Files Summary

```
Actuall/
├── compare_ligand_pocket_quantum_vs_classical.ipynb          # Original (GPU simulator)
├── compare_ligand_pocket_quantum_vs_classical_IBM.ipynb      # IBM Quantum version
├── IBM_QUANTUM_SETUP.md                                       # Setup guide
├── IBM_QUANTUM_README.md                                      # This file
├── requirements_ibm.txt                                       # IBM dependencies
├── ligand_pocket_qgnn/
│   ├── model.py                                               # Original model
│   ├── model_ibm.py                                           # IBM Quantum model
│   └── data.py                                                # Data processing
├── ligand_pocket_comparison_results/                          # Original results
└── ligand_pocket_comparison_results_IBM/                      # IBM results
```

## Model Architecture

Both versions use the same high-level architecture:

```
Input: Ligand Graph + Pocket Features
         ↓
    Ligand GNN Encoder → Ligand Vector (dim: N_QUBITS/2)
         ↓
    Pocket MLP Encoder → Pocket Vector (dim: N_QUBITS/2)
         ↓
    Concatenate → Combined Vector (dim: N_QUBITS)
         ↓
    Quantum Circuit (IBM Quantum) → Expectation value
         ↓
    Sigmoid → Binding Probability [0, 1]
```

The only difference is the quantum circuit execution backend.

## Quantum Circuit Details

```
Circuit Structure:
1. Angle Embedding: Encode classical data as qubit rotations
2. Variational Layers: StronglyEntanglingLayers ansatz
   - Each layer: RZ-RY-RZ rotations + CNOT entanglement
   - Number of layers: N_QLAYERS (default: 2)
3. Measurement: Expectation value of Pauli-Z on qubit 0
4. Shots: 8192 measurements per circuit evaluation
```

## Troubleshooting

### "No IBM Quantum account found"
→ Run the setup cell with your IBM token

### "Backend not available"
→ Check available backends at https://quantum.ibm.com/services/resources

### Training too slow
→ Switch to `ibmq_qasm_simulator` or reduce batch size

### Out of memory
→ Reduce `BATCH_SIZE` or use CPU instead of GPU for classical parts

## Support and Resources

- **IBM Quantum Platform**: https://quantum.ibm.com/
- **IBM Quantum Docs**: https://docs.quantum.ibm.com/
- **Qiskit Runtime**: https://qiskit.org/ecosystem/ibm-runtime/
- **PennyLane-Qiskit**: https://docs.pennylane.ai/projects/qiskit/

## Citation

If you use IBM Quantum hardware in your research, cite:

```bibtex
@misc{IBM-Quantum,
    title = {{IBM Quantum}},
    author = {{IBM}},
    url = {https://quantum.ibm.com/},
    year = {2024}
}
```

## Next Steps

1. Read [IBM_QUANTUM_SETUP.md](IBM_QUANTUM_SETUP.md) for detailed setup
2. Install dependencies: `pip install -r requirements_ibm.txt`
3. Setup your IBM Quantum account
4. Run the IBM notebook with simulator first
5. Test on real quantum hardware with reduced epochs
6. Compare results with original simulator version

---

**Created**: December 2024
**Status**: Ready to use
**Original**: Your quantum simulated training is still available in the original notebook
