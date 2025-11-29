# Migration Guide: Drug-Patient → Drug-Protein API

**Quick reference for updating your code to use the new drug-protein paradigm**

---

## Class Name Changes

| Old Name (Deprecated) | New Name | Status |
|----------------------|----------|--------|
| `PatientFeatures` | `ProteinPocketFeatures` | ✅ Aliased |
| `DrugPatientInteraction` | `DrugProteinInteraction` | ✅ Aliased |
| `DrugPatientDataProcessor` | `DrugProteinDataProcessor` | ✅ Aliased |
| `QuantumDrugPatientGNN` | `QuantumDrugProteinGNN` | ✅ Aliased |
| `DrugPatientTrainer` | `DrugProteinTrainer` | ✅ Aliased |

**Note:** All old names still work! They are aliases to the new implementations.

---

## Method Name Changes

### BipartiteGraph Methods

| Old Method | New Method | Status |
|-----------|------------|--------|
| `get_drug_features_matrix()` | `get_ligand_features_matrix()` | ✅ Both work |
| `get_patient_features_matrix()` | `get_pocket_features_matrix()` | ✅ Both work |
| `num_drugs()` | `num_ligands()` | ✅ Both work |
| `num_patients()` | `num_pockets()` | ✅ Both work |
| `add_patient(patient)` | `add_pocket(pocket)` | ⚠️ Use new method |

### DrugProteinDataProcessor Methods

| Old Method | New Method | Notes |
|-----------|------------|-------|
| `create_synthetic_patient_data()` | `create_synthetic_ligand_data()` | ⚠️ Signature changed |
| N/A | `load_protein_ligand_data()` | ✅ New method |

---

## Parameter Name Changes

### Model Constructor

```python
# Old way (still works)
model = QuantumDrugPatientGNN(
    drug_dim=19,
    patient_dim=41,
    num_qubits=4
)

# New way (recommended)
model = QuantumDrugProteinGNN(
    ligand_dim=19,
    pocket_dim=41,
    num_qubits=4
)
```

### Forward Pass

```python
# Old signature (still works)
output = model(drug_features, patient_features)

# New signature (same, just renamed variables)
output = model(ligand_features, pocket_features)
```

---

## Data Structure Changes

### Edge Features

**Old format** (5 features):
```python
[efficacy, dose, duration, outcome, num_adverse_events]
```

**New format** (4 features):
```python
[outcome, binding_affinity, distance, interaction_type_index]
```

**Impact:** If you directly access edge features by index, update your code:
```python
# Old
outcome = edge_features[:, 3]  # outcome was at index 3

# New
outcome = edge_features[:, 0]  # outcome now at index 0
```

---

## Step-by-Step Migration

### Step 1: Update Imports

```python
# Before
from drug_patient_qgnn import (
    DrugPatientDataProcessor,
    QuantumDrugPatientGNN,
    DrugPatientTrainer,
    PatientFeatures
)

# After
from drug_patient_qgnn import (
    DrugProteinDataProcessor,
    QuantumDrugProteinGNN,
    DrugProteinTrainer,
    ProteinPocketFeatures
)
```

### Step 2: Update Data Loading

```python
# Before
processor = DrugPatientDataProcessor(data_dir="data/")
processor.load_protein_ligand_data(max_samples=100)
processor.create_synthetic_patient_data(n_patients=200)
processor.create_synthetic_interactions()

# After
processor = DrugProteinDataProcessor(data_dir="data/")
processor.load_protein_ligand_data(max_samples=100)
processor.create_synthetic_ligand_data(n_ligands=100)
processor.create_synthetic_interactions()
```

### Step 3: Update Model Creation

```python
# Before
graph = processor.graph
model = QuantumDrugPatientGNN(
    drug_dim=graph.get_drug_features_matrix().shape[1],
    patient_dim=graph.get_patient_features_matrix().shape[1],
    num_qubits=4
)

# After
graph = processor.graph
model = QuantumDrugProteinGNN(
    ligand_dim=graph.get_ligand_features_matrix().shape[1],
    pocket_dim=graph.get_pocket_features_matrix().shape[1],
    num_qubits=4
)
```

### Step 4: Update Training

```python
# Before
trainer = DrugPatientTrainer(model, learning_rate=0.001)
trainer.fit(graph, epochs=100)

# After (same, just rename the class)
trainer = DrugProteinTrainer(model, learning_rate=0.001)
trainer.fit(graph, epochs=100)
```

### Step 5: Update Variable Names (Optional but Recommended)

```python
# Before
drug_features = ...
patient_features = ...
outcome = model(drug_features, patient_features)

# After
ligand_features = ...
pocket_features = ...
binding_prob = model(ligand_features, pocket_features)
```

---

## Common Migration Scenarios

### Scenario 1: Loading Real Data

```python
# Old approach
processor = DrugPatientDataProcessor(data_dir="pdb_data/")
processor.load_protein_ligand_data(max_samples=1000)
processor.create_synthetic_patient_data(n_patients=500)
processor.create_synthetic_interactions(interaction_rate=0.1)

# New approach (more accurate)
processor = DrugProteinDataProcessor(data_dir="pdb_data/")
processor.load_protein_ligand_data(max_samples=1000)
processor.create_synthetic_ligand_data(n_ligands=500)
processor.create_synthetic_interactions(
    interaction_rate=0.1,
    binding_rate=0.6  # probability of binding vs non-binding
)
```

### Scenario 2: Custom Pocket Features

```python
# Before
from drug_patient_qgnn import PatientFeatures
patient = PatientFeatures(
    patient_id="P001",
    age=45.0,
    sex=1,
    weight=75.0,
    ...
)
processor.graph.add_patient(patient)

# After
from drug_patient_qgnn import ProteinPocketFeatures
pocket = ProteinPocketFeatures(
    pocket_id="1a0n_A_pocket1",
    pocket_type="PLOC",
    volume=850.5,
    pmi1=120.3,
    pmi2=95.2,
    pmi3=80.1,
    ...
)
processor.graph.add_pocket(pocket)
```

### Scenario 3: Accessing Graph Statistics

```python
# Both old and new keys are available
stats = processor.get_statistics()

# Old keys (still work)
print(f"Drugs: {stats['num_drugs']}")
print(f"Patients: {stats['num_patients']}")
print(f"Drug dim: {stats['drug_feature_dim']}")
print(f"Patient dim: {stats['patient_feature_dim']}")

# New keys (recommended)
print(f"Ligands: {stats['num_ligands']}")
print(f"Pockets: {stats['num_pockets']}")
print(f"Ligand dim: {stats['ligand_feature_dim']}")
print(f"Pocket dim: {stats['pocket_feature_dim']}")
```

---

## Testing Your Migration

### 1. Check Imports

```python
# Test that all imports work
from drug_patient_qgnn import (
    DrugProteinDataProcessor,
    QuantumDrugProteinGNN,
    DrugProteinTrainer,
    ProteinPocketFeatures,
    LigandFeatures,
    DrugProteinInteraction
)
print("✓ New API imports successful")

# Test backward compatibility
from drug_patient_qgnn import (
    DrugPatientDataProcessor,
    QuantumDrugPatientGNN,
    DrugPatientTrainer
)
print("✓ Old API imports still work")
```

### 2. Verify Aliases

```python
assert DrugPatientDataProcessor is DrugProteinDataProcessor
assert QuantumDrugPatientGNN is QuantumDrugProteinGNN
assert DrugPatientTrainer is DrugProteinTrainer
print("✓ Backward compatibility aliases verified")
```

### 3. Test Data Loading

```python
processor = DrugProteinDataProcessor(data_dir="test_data/")
n_pockets = processor.load_protein_ligand_data(max_samples=10)
processor.create_synthetic_ligand_data(n_ligands=20)
processor.create_synthetic_interactions(interaction_rate=0.2)

assert processor.graph.num_pockets() == n_pockets
assert processor.graph.num_ligands() == 20
print("✓ Data loading works correctly")
```

### 4. Test Model

```python
model = QuantumDrugProteinGNN(
    ligand_dim=19,
    pocket_dim=19,
    num_qubits=4,
    use_quantum=False  # Use classical for faster testing
)

import torch
ligand_batch = torch.randn(8, 19)
pocket_batch = torch.randn(8, 19)
output = model(ligand_batch, pocket_batch)

assert output.shape == (8, 1)
assert (output >= 0).all() and (output <= 1).all()  # Sigmoid output
print("✓ Model forward pass works")
```

---

## Troubleshooting

### Issue: `AttributeError: 'BipartiteGraph' object has no attribute 'add_patient'`

**Solution:** Use `add_pocket()` instead:
```python
# Wrong
graph.add_patient(patient)

# Correct
graph.add_pocket(pocket)
```

### Issue: `TypeError: __init__() got an unexpected keyword argument 'patient_dim'`

**Solution:** The new model requires `pocket_dim` (but also accepts `patient_dim` for compatibility):
```python
# This works (new API)
model = QuantumDrugProteinGNN(ligand_dim=19, pocket_dim=19)

# This also works (backward compat)
model = QuantumDrugProteinGNN(drug_dim=19, patient_dim=19)
```

### Issue: Edge features index out of range

**Solution:** Edge feature format changed. Outcome is now at index 0 instead of 3:
```python
# Old code
outcome = edge_features[:, 3]

# New code
outcome = edge_features[:, 0]

# Or use the graph method
outcome = graph.get_edge_labels()
```

### Issue: `AttributeError: 'DrugProteinDataProcessor' object has no attribute 'create_synthetic_patient_data'`

**Solution:** Method renamed to `create_synthetic_ligand_data()`:
```python
# Old
processor.create_synthetic_patient_data(n_patients=100)

# New
processor.create_synthetic_ligand_data(n_ligands=100)
```

---

## Need Help?

1. **Check the examples:** See [run_pipeline.py](drug_patient_qgnn/examples/run_pipeline.py) for a complete working example
2. **Review the summary:** Read [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) for detailed changes
3. **Read the main README:** Updated [README.md](README.md) with new API documentation
4. **File an issue:** If you find bugs or have questions

---

## Deprecation Timeline

| Version | Status |
|---------|--------|
| **v0.2.x (current)** | Full backward compatibility. No warnings. |
| **v0.3.0 (planned)** | Deprecation warnings when old API is used. |
| **v1.0.0 (future)** | Old API removed. New API only. |

**Recommendation:** Migrate to the new API now to avoid issues in future releases!

---

**Happy coding!**
