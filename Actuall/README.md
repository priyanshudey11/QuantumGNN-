# Drug-Patient Interaction Prediction with Quantum GNN

A production-quality implementation of a **Bipartite Quantum Graph Neural Network** for predicting drug-patient interactions. This system combines molecular features from protein-ligand binding data with patient clinical characteristics to predict treatment outcomes.

## 🌟 Key Features

- ⚛️ **Quantum Graph Neural Network** with PennyLane
- 🔬 **Bipartite Graph Structure**: Drug nodes + Patient nodes + Interaction edges
- 🧬 **Real Molecular Data**: Loads 3D descriptors from PDB protein-ligand structures
- 🏥 **Patient Features**: Demographics, genetics, comorbidities, lab values
- 📊 **Production-Ready**: Complete training pipeline with metrics, visualization, and checkpointing
- 🔄 **Classical Baseline**: Optional classical mode for comparison

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install torch numpy pandas scikit-learn pennylane matplotlib

# Optional: molecular processing
pip install rdkit biopython

# Install package
pip install -e .
```

### Basic Usage

```python
from drug_patient_qgnn import (
    DrugPatientDataProcessor,
    QuantumDrugPatientGNN,
    DrugPatientTrainer
)

# 1. Load drug data from PDB descriptors
processor = DrugPatientDataProcessor(
    data_dir="/media/priyanshu/SD/othercode/data"
)
processor.load_protein_ligand_data(max_samples=100)

# 2. Generate patient data
processor.create_synthetic_patient_data(n_patients=200)
processor.create_synthetic_interactions(interaction_rate=0.05)

# 3. Create quantum GNN model
graph = processor.graph
model = QuantumDrugPatientGNN(
    drug_dim=graph.get_drug_features_matrix().shape[1],
    patient_dim=graph.get_patient_features_matrix().shape[1],
    num_qubits=4,
    num_qlayers=2,
    use_quantum=True
)

# 4. Train
trainer = DrugPatientTrainer(model, learning_rate=0.001)
trainer.fit(graph, epochs=100, val_split=0.2)

# 5. Predict
drug_features, patient_features = ...  # Your data
probs, preds = model.predict_batch(drug_features, patient_features)
```

### Command Line

```bash
# Run full pipeline
python drug_patient_qgnn/examples/run_pipeline.py \
    --epochs 100 \
    --batch_size 32 \
    --num_qubits 4 \
    --num_qlayers 2 \
    --use_quantum True

# Classical baseline
python drug_patient_qgnn/examples/run_pipeline.py --use_quantum False

# Custom data path
python drug_patient_qgnn/examples/run_pipeline.py \
    --data_dir /path/to/your/data \
    --n_patients 500
```

### Jupyter Notebook Tutorial

```bash
jupyter notebook tutorial.ipynb
```

The tutorial includes:
- Data loading and exploration
- Model training (quantum & classical)
- Performance visualization
- Quantum vs classical comparison
- Prediction examples

## 📁 Project Structure

```
.
├── drug_patient_qgnn/          # Main package
│   ├── __init__.py             # API exports
│   ├── data_processing.py      # Graph structures & data loading
│   ├── model.py                # Quantum GNN implementation
│   ├── trainer.py              # Training loop & metrics
│   ├── utils.py                # Utilities & visualization
│   └── examples/
│       └── run_pipeline.py     # Complete example script
│
├── tests/                      # Test suite
│   ├── test_graph.py           # Graph construction tests
│   ├── test_model_forward.py   # Model forward pass tests
│   └── test_trainer_loop.py    # Training loop tests
│
├── tutorial.ipynb              # Jupyter tutorial
├── pyproject.toml              # Package configuration
│
├── DRUG_PATIENT_PIPELINE_README.md  # Detailed documentation
├── DATASET_NOTES.md                 # Dataset explanation
├── IMPLEMENTATION_SUMMARY.md        # Implementation details
└── README.md                        # This file
```

## 🎯 Model Architecture

### Bipartite Graph

```
Drugs (D1, D2, ..., Dn)    Patients (P1, P2, ..., Pm)
      │                           │
      ├───────────┬───────────────┤
      │           │               │
   Edge(D1,P1) Edge(D1,P2)   Edge(D2,P1)
   [features]  [features]    [features]
```

- **Drug Nodes**: 3D molecular descriptors (19 features)
- **Patient Nodes**: Clinical/demographic features (30+ features)
- **Edges**: Observed interactions with outcomes

### Quantum GNN

```
Drug Encoder → Quantum Interaction Layer → Output Head
              ↑
Patient Encoder
```

**Mathematical Formulation:**

1. **Classical Pre-embedding:**
   ```
   h_D = σ(W_D × x_D + b_D)
   h_P = σ(W_P × x_P + b_P)
   ```

2. **Quantum Encoding:**
   ```
   |ψ_D⟩ = ⊗_k R_y(h_D[k])|0⟩
   |ψ_P⟩ = ⊗_k R_y(h_P[k])|0⟩
   ```

3. **Entangling Ansatz:**
   - CNOT gates between drug and patient qubits
   - Variational layers with R_z, R_x rotations
   - Ring topology for full connectivity

4. **Measurement & Output:**
   ```
   z = ⟨ψ_final | Z | ψ_final⟩
   outcome_prob = σ(MLP(z))
   ```

## 📊 Dataset

This implementation works with the **Protein-Protein Interaction Dataset** from Nature Scientific Data (2024):

- **23,000+ binding pockets** from protein-protein interactions
- **3,700+ proteins** across 500+ organisms
- **~3,500 ligands** with binding information
- **3D molecular descriptors** from VolSite analysis

### Drug Features (from PDB descriptors):
- Volume, shape descriptors (PMI, NPR, Rgyr)
- Geometric properties (Asphericity, Eccentricity)
- Atom type counts (C, N, O, etc.)

### Patient Features (synthetic or real EHR):
- Demographics: age, sex, weight
- Genetic markers
- Comorbidities
- Lab values
- Medication history

See `DATASET_NOTES.md` for detailed information.

## 🧪 Testing

Run the test suite:

```bash
# All tests
python tests/test_graph.py
python tests/test_model_forward.py
python tests/test_trainer_loop.py

# Or with pytest
pytest tests/ -v
```

## 📈 Performance

The model tracks multiple metrics:
- **Loss**: Binary cross-entropy
- **Accuracy**: Classification accuracy
- **AUC-ROC**: Area under ROC curve
- **Precision/Recall/F1**: From confusion matrix

Example training output:
```
Epoch 50/100 - train_loss: 0.4523 - train_acc: 0.7812 - val_loss: 0.4891 - val_acc: 0.7654 - val_auc: 0.8234
```

## 🔧 Configuration

### Model Parameters

```python
QuantumDrugPatientGNN(
    drug_dim=19,           # From molecular descriptors
    patient_dim=30,        # From patient features
    num_qubits=4,          # Qubits per side (8 total)
    num_qlayers=2,         # Variational layers
    hidden_dim=64,         # Encoder hidden size
    use_quantum=True       # True: quantum, False: classical
)
```

### Training Parameters

```python
trainer.fit(
    graph,
    epochs=100,            # Training epochs
    batch_size=32,         # Batch size
    val_split=0.2,         # Validation fraction
    verbose=2              # Verbosity level
)
```

## 💡 Advanced Usage

### Custom Patient Data

```python
from drug_patient_qgnn import PatientFeatures
import numpy as np

# Create patient from real data
patient = PatientFeatures(
    patient_id="P001",
    age=45.0,
    sex=1,  # 0=Female, 1=Male
    weight=75.5,
    genetic_markers=np.array([0.1, 0.2, ...]),
    comorbidities=np.array([1, 0, 1, 0, 0]),
    lab_values=np.array([100, 1.1, 30, 25, ...]),
    prior_medications=np.array([1, 0, 0, 1, 0])
)

processor.graph.add_patient(patient)
```

### Quantum Device Configuration

```python
# Default: CPU simulator
model = QuantumDrugPatientGNN(..., use_quantum=True)

# For specific PennyLane device
import pennylane as qml
device = qml.device('lightning.gpu', wires=8)  # GPU-accelerated
# Then modify model.py to use this device
```

### Model Interpretation

```python
# Feature importance (gradient-based)
drug_features.requires_grad = True
output = model(drug_features, patient_features)
output.backward()
importance = drug_features.grad.abs()

# Quantum circuit visualization
import pennylane as qml
fig, ax = qml.draw_mpl(model.interaction_layer.qnode)(...)
```

## 🤝 Contributing

This is a research prototype. To extend:

1. **Add new features**: Modify `PatientFeatures` in `data_processing.py`
2. **Custom quantum circuits**: Edit `QuantumInteractionLayer` in `model.py`
3. **New metrics**: Extend `calculate_metrics` in `utils.py`
4. **Different architectures**: Create new model classes

## 📚 Documentation

- **Main README**: Comprehensive guide (this file)
- **DRUG_PATIENT_PIPELINE_README.md**: Full pipeline documentation
- **DATASET_NOTES.md**: Dataset structure and features
- **IMPLEMENTATION_SUMMARY.md**: Technical implementation details
- **tutorial.ipynb**: Interactive walkthrough

## 📄 Citation

If you use this code, please cite:

```bibtex
@software{drug_patient_qgnn_2025,
  title={Drug-Patient Interaction Pipeline with Quantum GNN},
  author={Drug-Patient QGNN Team},
  year={2025},
  url={https://github.com/your-repo/drug-patient-qgnn}
}
```

Dataset citation:
```bibtex
@article{moine2024comprehensive,
  title={A comprehensive dataset of protein-protein interactions and ligand binding pockets for advancing drug discovery},
  author={Moine-Franel, Alexandra and others},
  journal={Scientific Data},
  volume={11},
  number={402},
  year={2024},
  doi={10.1038/s41597-024-03233-z}
}
```

## ⚠️ Disclaimer

This is a research prototype for educational and research purposes. **Consult medical professionals and follow regulatory guidelines before using in clinical settings.**

## 📞 Support

- Report issues on GitHub
- Check PennyLane docs: https://docs.pennylane.ai/
- Review dataset paper: https://doi.org/10.1038/s41597-024-03233-z

## 📜 License

MIT License - See LICENSE file for details

---

**Status**: Production-ready ✅
**Python**: 3.8+
**PyTorch**: 2.0+
**PennyLane**: 0.32+

Built with ⚛️ quantum computing for 🧬 drug discovery
