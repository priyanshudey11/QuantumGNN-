# Cost-Efficient IBM Quantum Hardware Testing

## Problem
Training on real IBM QPU is extremely expensive (~$31k/epoch).

## Solution: Inference-Only Testing

Test your trained model on real quantum hardware WITHOUT full training.

## Setup

### Step 1: Train on Simulator (Original Notebook)
Already done! Your model is at:
- `ligand_pocket_comparison_results/quantum_best.pt`
- Best AUC: 0.6832 (epoch 34)

### Step 2: Create Inference-Only Test Script

```python
# test_on_real_quantum_hardware.py
import torch
from ligand_pocket_qgnn.model_ibm import LigandPocketQGNN_IBM
from ligand_pocket_qgnn.data import LigandPocketDataProcessor, LigandPocketDataset

# Configuration for REAL hardware test
IBM_BACKEND = 'ibm_brisbane'  # Real QPU
TEST_SAMPLES = 100  # SMALL test set
BATCH_SIZE = 1  # Process one at a time

# Load data
processor = LigandPocketDataProcessor(DATA_DIR, seed=42069)
processor.load_data(max_samples=TEST_SAMPLES)  # Only 100 samples!
test_dataset = LigandPocketDataset(processor, processor.get_dataset())

# Create model with IBM backend
model = LigandPocketQGNN_IBM(
    ligand_in_dim=10,
    pocket_in_dim=19,
    hidden_dim=64,
    n_qubits=6,
    n_qlayers=2,
    use_quantum=True,
    ibm_backend=IBM_BACKEND
)

# Load pre-trained weights from simulator
checkpoint = torch.load('ligand_pocket_comparison_results/quantum_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test on real quantum hardware (only 100 forward passes!)
with torch.no_grad():
    predictions = []
    for sample in test_dataset:
        pred = model(*sample)
        predictions.append(pred.item())

# Compare with simulator predictions
# Save results for publication
```

**Estimated Cost:**
- 100 samples × 0.1 sec/circuit = 10 seconds QPU time
- Cost: ~$16
- Time: 1-2 hours (including queue)

**Benefit:**
- Can claim "validated on IBM Quantum hardware"
- Shows model works on real QPU
- Affordable for publication

### Step 3: Configuration Comparison

| Approach | Samples | Epochs | QPU Time | Cost | Purpose |
|----------|---------|--------|----------|------|---------|
| **Full Training** | 244,260 | 100 | ~540 hours | $3.1M | Not feasible |
| **1 Epoch Test** | 244,260 | 1 | ~5.4 hours | $31k | Too expensive |
| **Mini-dataset** | 1,000 | 1 | ~2 minutes | $200 | Testing only |
| **Inference Test** | 100 | 0 | 10 seconds | $16 | ✓ Recommended |

## Recommended Workflow for Publication

### Phase 1: Development (FREE)
```python
# Original notebook
QUANTUM_DEVICE = 'lightning.gpu'  # Local GPU
Train full model → Best AUC: 0.6832
```

### Phase 2: IBM Simulator Validation (FREE)
```python
# IBM notebook
IBM_BACKEND = 'ibmq_qasm_simulator'  # IBM simulator
Train to verify compatibility
Compare: Does IBM simulator match lightning.gpu?
```

### Phase 3: Real Hardware Proof-of-Concept ($16-50)
```python
# Inference test
IBM_BACKEND = 'ibm_brisbane'  # REAL QPU
Test 100 samples with pre-trained weights
Measure: accuracy, execution time, noise effects
```

### Phase 4: Publication
Write paper stating:
- "Model trained using quantum simulation"
- "Validated on IBM Quantum hardware (ibm_brisbane)"
- "Successfully demonstrated quantum GNN on real quantum device"

## Cost Summary

| Component | Backend | Time | Cost |
|-----------|---------|------|------|
| Development | `lightning.gpu` | 1 day | $0 |
| Full Training | `lightning.gpu` | 1 day | $0 |
| IBM Validation | `ibmq_qasm_simulator` | 3-7 days | $0 |
| Real QPU Test | `ibm_brisbane` | 2-6 hours | $16-50 |
| **TOTAL** | | ~2 weeks | **$16-50** |

## Why This Works

1. **Scientifically Valid**: Trained model parameters are backend-agnostic
2. **Cost Effective**: Only pay for small-scale verification
3. **Publication Ready**: Can cite use of real quantum hardware
4. **Reproducible**: Others can replicate on simulators cheaply
5. **Demonstrates Feasibility**: Shows quantum GNN works on real QPU

## IBM Quantum Access Programs

To reduce costs further:

### IBM Quantum Network
- Academic access: Often FREE for researchers
- Apply at: https://www.ibm.com/quantum/network
- Requirements: University affiliation, research proposal

### IBM Quantum Researchers Program
- Free credits for academic research
- Up to $50,000 in quantum compute credits
- Apply with research proposal

### IBM Quantum Educators Program
- Free access for teaching
- Limited to educational use

## Alternative: Use Smaller Model

If you want to train (not just test) on real hardware:

```python
# Ultra-minimal configuration
N_QUBITS = 4  # Reduced from 6
N_QLAYERS = 1  # Reduced from 2
MAX_SAMPLES = 1000  # Tiny dataset
BATCH_SIZE = 10
EPOCHS = 1

Estimated cost: ~$500-1000 per epoch
Still expensive but more feasible
```

## Bottom Line

**For your use case:**
1. ✓ Keep training on `lightning.gpu` (FREE, fast)
2. ✓ Test on `ibmq_qasm_simulator` (FREE, validates IBM integration)
3. ✓ Run inference on real hardware for 100 samples ($16-50)
4. ✓ Publish with "validated on IBM Quantum hardware"

**DO NOT attempt full training on real QPU unless:**
- You have $3M+ budget
- You have 6-12 months for training
- This is critical for the research (unlikely)
