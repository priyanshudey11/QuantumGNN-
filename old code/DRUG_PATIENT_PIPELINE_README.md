# Drug-Patient Interaction Pipeline with Quantum GNN

A comprehensive pipeline for predicting drug-patient interactions using quantum graph neural networks (QGNN). This system combines molecular drug features from protein-ligand binding data with patient characteristics to predict treatment outcomes.

## Overview

### What This Pipeline Does

1. **Drug Representation**: Encodes drugs using 3D binding pocket descriptors from your PDB protein-ligand data
2. **Patient Representation**: Encodes patient features (demographics, genetics, clinical data)
3. **Interaction Modeling**: Uses quantum circuits to learn complex drug-patient interaction patterns
4. **Outcome Prediction**: Predicts treatment efficacy and adverse events

### Architecture

```
Drug Features          Patient Features
     │                       │
     ├─ Drug Encoder        ├─ Patient Encoder
     │  (Classical NN)      │  (Classical NN)
     │                      │
     └──────┬───────────────┘
            │
     Quantum Interaction Layer
     (Entangled qubits encode drug-patient interactions)
            │
     Output Layer
     (Efficacy/Outcome prediction)
```

## Data Structure

### Input Data Sources

#### 1. Drug Data (From Your PDB Files)

Your existing protein-ligand binding data in `othercode/data/`:

```
data/
├── 1a0n/
│   ├── pdb1a0n.ent                                    # Protein structure
│   ├── 1a0n--AB--P27986--P06241.pdb                  # Protein-protein complex
│   ├── 1a0n--AB--P27986__interface-residues_6A.txt   # Interface residues
│   └── results/
│       └── 1a0n--A--P27986__Repair-H_descriptors_3d.csv  # 3D descriptors
├── 8exo/
│   ├── 8exo--A--P42336--X3W-1101.pdb                 # Protein-ligand complex
│   └── ...
```

**3D Descriptor Features** (from CSV files):
- `Volume`: Binding pocket volume
- `PMI1, PMI2, PMI3`: Principal moments of inertia (shape)
- `NPR1, NPR2`: Normalized principal moment ratios
- `Rgyr`: Radius of gyration
- `Asphericity, SpherocityIndex, Eccentricity`: Shape descriptors
- `InertialShapeFactor`: Rotational properties
- `CZ, CA, O, OD1, OG, N, NZ, DU`: Atom type counts

#### 2. Patient Data (To Be Provided)

Expected patient data format:

```python
PatientFeatures:
  - patient_id: str
  - age: float (18-100)
  - sex: int (0=Female, 1=Male)
  - weight: float (kg)
  - genetic_markers: array (SNPs, gene expression)
  - comorbidities: array (diabetes, hypertension, etc.)
  - lab_values: array (blood tests, biomarkers)
  - prior_medications: array (medication history)
```

**Recommended sources:**
- Electronic Health Records (EHR)
- Clinical trial databases
- Genomic databases (1000 Genomes, UK Biobank)
- FDA Adverse Event Reporting System (FAERS)

#### 3. Interaction Data

```python
DrugPatientInteraction:
  - drug_id: str (ligand identifier)
  - patient_id: str
  - efficacy: float (0-1 scale)
  - adverse_events: List[str]
  - dose: float (mg)
  - duration: float (days)
  - outcome: int (0=failure, 1=success)
```

## Graph Representation

### Bipartite Graph Structure

```
Drugs (D1, D2, ..., Dn)    Patients (P1, P2, ..., Pm)
      │                           │
      ├───────────┬───────────────┤
      │           │               │
   Edge(D1,P1) Edge(D1,P2)   Edge(D2,P1)
   [features]  [features]    [features]
```

- **Drug Nodes**: Molecular/binding features (from PDB descriptors)
- **Patient Nodes**: Clinical/demographic features
- **Edges**: Observed or potential interactions
- **Edge Features**: Dose, duration, efficacy, outcome

### Quantum Encoding

The quantum circuit encodes drug-patient interactions:

1. **Drug qubits** (first half): Encode molecular features using RY rotations
2. **Patient qubits** (second half): Encode patient features using RY rotations
3. **Entanglement**: CNOT gates between drug and patient qubits model interactions
4. **Variational layers**: Learn interaction patterns
5. **Measurement**: PauliZ expectation values predict outcome

## Installation

### Requirements

```bash
# Core dependencies
pip install torch numpy pandas

# Quantum computing
pip install pennylane

# Optional: GPU acceleration
pip install pennylane-lightning-gpu  # NVIDIA CUDA
# OR
pip install pennylane-lightning-mtl  # Apple Metal (M1/M2)

# Optional: For molecular processing
pip install rdkit biopython
```

### Setup

```bash
cd /Users/priyanshudey/Code/Qunatum
python drug_patient_quantum_gnn_pipeline.py
```

## Usage

### Basic Example

```python
from drug_patient_quantum_gnn_pipeline import (
    DrugPatientDataProcessor,
    QuantumDrugPatientGNN,
    DrugPatientTrainer
)

# 1. Load and process data
processor = DrugPatientDataProcessor(data_dir="othercode/data")

# Load drug features from your PDB descriptors
processor.load_protein_ligand_data(max_samples=100)

# Add patient data (real or synthetic)
processor.create_synthetic_patient_data(n_patients=200)

# Create interactions
processor.create_synthetic_interactions(interaction_rate=0.05)

# 2. Create model
graph = processor.graph
model = QuantumDrugPatientGNN(
    drug_dim=graph.get_drug_features_matrix().shape[1],
    patient_dim=graph.get_patient_features_matrix().shape[1],
    num_qubits=8,
    num_qlayers=2,
    use_quantum=True
)

# 3. Train
trainer = DrugPatientTrainer(model, learning_rate=0.001)
trainer.fit(graph, epochs=100, val_split=0.2)

# 4. Save model
torch.save(model.state_dict(), "trained_model.pt")
```

### Advanced: Custom Patient Data

```python
from drug_patient_quantum_gnn_pipeline import PatientFeatures
import pandas as pd

# Load real patient data
patient_df = pd.read_csv("your_patient_data.csv")

for _, row in patient_df.iterrows():
    patient = PatientFeatures(
        patient_id=row['patient_id'],
        age=row['age'],
        sex=1 if row['sex'] == 'M' else 0,
        weight=row['weight_kg'],
        genetic_markers=np.array(row['snp_profile'].split(',')).astype(float),
        comorbidities=np.array([
            row['diabetes'],
            row['hypertension'],
            row['heart_disease'],
            row['kidney_disease'],
            row['liver_disease']
        ]),
        lab_values=np.array([
            row['glucose'],
            row['creatinine'],
            row['ast'],
            row['alt'],
            row['ldl'],
            row['hdl'],
            row['triglycerides'],
            row['hemoglobin']
        ]),
        prior_medications=np.array(row['medication_history'].split(',')).astype(int)
    )
    processor.graph.add_patient(patient)
```

### Prediction on New Drug-Patient Pairs

```python
# Load trained model
model = QuantumDrugPatientGNN(drug_dim, patient_dim, num_qubits=8)
model.load_state_dict(torch.load("trained_model.pt"))
model.eval()

# Prepare features
drug_features = torch.tensor(drug_vector).unsqueeze(0)  # (1, drug_dim)
patient_features = torch.tensor(patient_vector).unsqueeze(0)  # (1, patient_dim)

# Predict
with torch.no_grad():
    outcome_prob = model(drug_features, patient_features)
    print(f"Predicted success probability: {outcome_prob.item():.2%}")
```

## Pipeline Components

### 1. DrugPatientDataProcessor

**Purpose**: Load and preprocess drug-patient data

**Key Methods**:
- `load_protein_ligand_data()`: Loads 3D descriptors from your PDB data
- `create_synthetic_patient_data()`: Generate synthetic patients (for testing)
- `create_synthetic_interactions()`: Generate synthetic interactions
- `save_graph()` / `load_graph()`: Serialize processed data

### 2. QuantumDrugPatientGNN

**Purpose**: Quantum neural network model

**Architecture**:
- Drug encoder: Linear(drug_dim → 64) → ReLU → Linear(64 → num_qubits)
- Patient encoder: Linear(patient_dim → 64) → ReLU → Linear(64 → num_qubits)
- Quantum layer: Variational quantum circuit with entanglement
- Output head: Linear(num_qubits → 32) → ReLU → Linear(32 → 1) → Sigmoid

**Quantum Circuit**:
```
Drug Qubits:    |0⟩ ─ RY(θ₁) ─ ┬ ─── Variational Layers ─── Z
                |0⟩ ─ RY(θ₂) ─ │
                                ×
Patient Qubits: |0⟩ ─ RY(θ₃) ─ ┴ ─── Variational Layers ─── Z
                |0⟩ ─ RY(θ₄) ─

θ: Feature encodings
┬┴: CNOT (entanglement between drug and patient)
×: Ring topology CNOTs
Z: PauliZ measurements
```

### 3. DrugPatientTrainer

**Purpose**: Training and evaluation

**Key Methods**:
- `train_epoch()`: Single training epoch
- `evaluate()`: Compute validation metrics
- `fit()`: Full training loop with train/val split

**Metrics**:
- Binary cross-entropy loss
- Prediction accuracy
- (Can extend to: AUC-ROC, precision, recall, F1)

## How to Map Your Data

### Step 1: Extract Drug Features from PDB Data

Your CSV files contain binding pocket descriptors. The pipeline automatically extracts:

```python
# From: data/1a0n/results/1a0n--A--P27986__Repair-H_descriptors_3d.csv
# Extracts: Volume, PMI1-3, NPR1-2, Rgyr, Asphericity, etc.

processor.load_protein_ligand_data()
# This reads all *_descriptors_3d.csv files
```

### Step 2: Prepare Patient Data

Create a CSV file with patient information:

```csv
patient_id,age,sex,weight_kg,diabetes,hypertension,heart_disease,glucose,creatinine,snp_profile
P0001,45,M,75.5,0,1,0,95.2,1.1,0.2|0.8|0.5|...
P0002,62,F,68.0,1,1,1,142.5,1.8,0.1|0.3|0.7|...
```

### Step 3: Prepare Interaction Data

Create interaction records:

```csv
drug_id,patient_id,efficacy,dose_mg,duration_days,outcome,adverse_events
#3hwe--A--P80188__Repair-H.pdb.mol2_1,P0001,0.85,100,30,1,"nausea,headache"
#3hwe--A--P80188__Repair-H.pdb.mol2_1,P0002,0.42,100,14,0,"dizziness,fatigue"
```

### Step 4: Run Pipeline

```bash
python drug_patient_quantum_gnn_pipeline.py
```

## Key Advantages of Quantum GNN

1. **Entanglement**: Captures complex non-linear drug-patient interactions
2. **Superposition**: Explores multiple interaction pathways simultaneously
3. **Quantum Advantage**: May provide computational speedup for large datasets
4. **Feature Interactions**: Naturally models higher-order feature interactions

## Model Interpretability

To understand why the model makes predictions:

```python
# 1. Feature importance (gradient-based)
drug_features.requires_grad = True
output = model(drug_features, patient_features)
output.backward()
drug_importance = drug_features.grad.abs()

# 2. Quantum circuit visualization
import pennylane as qml
fig, ax = qml.draw_mpl(model.quantum_layer)(inputs, weights)
fig.savefig("quantum_circuit.png")

# 3. Attention visualization
# (Can extend model to include attention mechanisms)
```

## Performance Optimization

### GPU Acceleration

```python
# Use CUDA (NVIDIA)
model = QuantumDrugPatientGNN(...).to('cuda')
trainer = DrugPatientTrainer(model, device='cuda')

# Or Metal (Apple M1/M2)
model = QuantumDrugPatientGNN(...).to('mps')
trainer = DrugPatientTrainer(model, device='mps')
```

### Quantum Device Options

```python
# Fast CPU simulator
device = qml.device('default.qubit', wires=8)

# GPU-accelerated (if installed)
device = qml.device('lightning.gpu', wires=8)

# Real quantum hardware (IBM Quantum)
device = qml.device('qiskit.ibmq', wires=8, backend='ibmq_manila')
```

## Future Extensions

### 1. Multi-Task Learning

Predict multiple outcomes simultaneously:
- Efficacy score
- Adverse event probability
- Optimal dosage
- Treatment duration

### 2. Graph Attention

Add attention mechanism to weight important features:

```python
# Drug-patient attention
attention_scores = softmax(Q @ K^T / sqrt(d_k))
weighted_features = attention_scores @ V
```

### 3. Temporal Modeling

Track patient response over time:

```python
# Add LSTM/GRU for sequential data
temporal_encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2)
```

### 4. Explainable AI

Integrate SHAP or LIME for model explanations:

```python
import shap
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_data)
```

## Validation Strategy

### Cross-Validation

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True)
for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
    model = QuantumDrugPatientGNN(...)
    trainer = DrugPatientTrainer(model)
    # Train on fold...
```

### External Validation

- Test on held-out hospitals/populations
- Temporal validation (train on past data, test on future)
- Cross-dataset validation (train on Dataset A, test on Dataset B)

## Troubleshooting

### Issue: "PennyLane not installed"

```bash
pip install pennylane
```

### Issue: "CUDA out of memory"

Reduce batch size or use gradient accumulation:

```python
# Train with smaller batches
trainer.fit(graph, epochs=100, batch_size=16)
```

### Issue: "Quantum circuit too slow"

- Reduce `num_qubits` (try 4-6 instead of 8)
- Reduce `num_qlayers` (try 1 instead of 2)
- Use classical fallback: `use_quantum=False`

## References

- PennyLane: https://pennylane.ai/
- Protein Data Bank: https://www.rcsb.org/
- Quantum Machine Learning: https://arxiv.org/abs/2101.11020

## Citation

If you use this pipeline, please cite:

```bibtex
@software{drug_patient_qgnn_2025,
  title={Drug-Patient Interaction Pipeline with Quantum GNN},
  author={Generated with Claude Code},
  year={2025},
  url={https://github.com/your-repo}
}
```

## Contact

For questions or issues:
- Open an issue on GitHub
- Check PennyLane documentation: https://docs.pennylane.ai/

---

Generated with Claude Code | 2025-11-03
