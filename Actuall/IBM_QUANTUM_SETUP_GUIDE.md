# IBM Quantum Setup Guide

## Current Status

### ✅ What's Working
- IBM Quantum account is configured and saved
- Available backends: `ibm_fez` (156 qubits), `ibm_torino` (133 qubits)
- Local quantum simulators are working: `lightning.qubit`, `default.qubit`

### ❌ Current Issue
**PennyLane compatibility problem**: PennyLane 0.43.1 has deprecated `QubitDevice`, but the `pennylane-qiskit` plugin (required for IBM backends) still tries to import it, causing an `ImportError`.

## Solutions

### Option 1: Use Local Simulators (CURRENT SETUP - RECOMMENDED FOR TESTING)

**What it does**: Simulates quantum circuits on your local computer

**Pros**:
- ✅ Fast (minutes per epoch)
- ✅ Free, unlimited usage
- ✅ Works immediately
- ✅ Good for testing and development

**Cons**:
- ❌ Not real quantum hardware
- ❌ No real quantum noise/effects

**Setup**: Already configured! Just run the notebook with:
```python
BACKEND = 'lightning.qubit'  # Already set in cell-6
```

### Option 2: Downgrade to Use IBM Quantum Hardware

**What it does**: Uses actual IBM quantum computers (ibm_fez, ibm_torino)

**Steps**:
1. Downgrade PennyLane and install the Qiskit plugin:
   ```bash
   pip install pennylane==0.32.0 pennylane-qiskit
   ```

2. In the notebook, change cell-1 import:
   ```python
   from ligand_pocket_qgnn.model_ibm import LigandPocketQGNN_IBM  # Use original
   ```

3. In cell-6, set:
   ```python
   IBM_BACKEND = 'ibm_fez'  # Real quantum hardware!
   ```

**Pros**:
- ✅ Real quantum computer!
- ✅ Actual quantum effects and noise
- ✅ Publishable results

**Cons**:
- ❌ VERY slow (estimated 611+ hours for full training)
- ❌ Queue times
- ❌ Requires downgrading PennyLane
- ❌ Limited free tier credits

## Recommendation

**For Development/Testing**:
- Use **Option 1** (local simulators) with `lightning.qubit`
- Fast iteration, test your code, debug

**For Final Experiments/Publication**:
- Use **Option 2** (real quantum hardware) with `ibm_fez`
- Reduce `EPOCHS` to 3-5 for feasibility
- Use small `BATCH_SIZE = 16`
- Run overnight/over weekend

## Current Notebook Configuration

The notebook is now configured to use **local simulators**:
- Cell 1: Imports `model_ibm_fixed.py` (compatible with PennyLane 0.43.1)
- Cell 6: `BACKEND = 'lightning.qubit'` (fast local simulator)
- Cell 8: `BATCH_SIZE = 256` (optimized for local simulation)

You can run the notebook as-is for quantum simulation training!

## Hardware Specifications

### Your Available IBM Quantum Backends
- **ibm_fez**: 156 qubits, Heron r2 processor, ONLINE, 0 pending jobs
- **ibm_torino**: 133 qubits, ONLINE
- **ibm_marrakesh**: 156 qubits, MAINTENANCE (not available)

### Recommended Settings for Real Hardware
```python
N_QUBITS = 6
N_QLAYERS = 2
BATCH_SIZE = 16
EPOCHS = 3  # Start small!
```

Estimated time per epoch on ibm_fez: ~20-40 hours (conservative)
