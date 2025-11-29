# Drug–Protein Interaction Pipeline with Quantum GNN

A comprehensive pipeline for predicting **drug–protein interactions** using Quantum Graph Neural Networks (QGNN).
This system combines **ligand (drug) features** and **protein pocket descriptors** from a large protein–protein / protein–ligand dataset to model and predict how drugs behave across different protein targets.

---

## Overview

### What This Pipeline Does

1. **Drug Representation**
   Encodes ligands (drugs) using molecular descriptors (from ligand structures) and/or pocket-level binding context.

2. **Protein Representation**
   Encodes protein binding pockets using 3D geometric and atom-type descriptors from the Nature Scientific Data dataset.

3. **Interaction Modeling**
   Uses a quantum-inspired interaction layer (QGNN) where **entangled qubits encode ligand–protein interactions**.

4. **Outcome Prediction**
   Predicts:
   - interaction type (competitive / non-competitive / allosteric), or
   - binding vs non-binding (binary), or
   - other task-specific labels you define.

---

## Architecture

```text
Ligand Features          Protein Pocket Features
      │                             │
      ├─ Ligand Encoder             ├─ Pocket Encoder
      │  (Classical NN)             │  (Classical NN)
      │                             │
      └──────────┬──────────────────┘
                 │
      Quantum Interaction Layer
      (Entangled qubits encode ligand–protein interactions)
                 │
           Output Layer
      (Binding / interaction prediction)
```

---

## Dataset: Protein–Protein / Protein–Ligand Interactions

This pipeline uses the dataset from:

> **"A comprehensive dataset of protein–protein interactions and ligand binding pockets for advancing drug discovery"**
> Nature Scientific Data, 2024.

### Dataset Statistics

* **23,000+ pockets** from protein–protein interactions
* **3,700+ proteins** across 500+ organisms
* **1,700+ protein families** (Pfam domains)
* **~3,500 ligands** with binding information

### Dataset Subsets

1. **HD (Heterodimer) Dataset**

   * 4,770 PDB structures
   * 18,266 pockets
   * Protein–protein interaction sites
   * Use as **negative** examples (no small-molecule ligand).

2. **PLOC (Protein–Ligand Orthosteric Competitive)**

   * 1,863 PDB structures
   * 2,325 pockets
   * 1,647 ligands
   * Ligands **compete** with protein partner at the interface
   * Ideal positive examples for drug discovery tasks.

3. **PLONC (Orthosteric Non-Competitive)**

   * 715 PDB structures
   * 817 pockets
   * 539 ligands
   * Ligands at interface but **non-competitive**.

4. **PLA (Allosteric)**

   * 1,613 PDB structures
   * 1,830 pockets
   * 1,277 ligands
   * Ligands **away from** the interface (allosteric modulators).

---

## Data Structure on Disk

Each PDB entry (e.g., `1a0n`) has:

```text
1a0n/
├── pdb1a0n.ent                                    # Raw PDB file
├── 1a0n--AB--P27986--P06241.pdb                  # Protein–protein complex
├── 1a0n--AB--P27986__interface-residues_6A.txt   # Interface residues
└── results/
    └── 1a0n--A--P27986__Repair-H_descriptors_3d.csv  # 3D pocket descriptors
```

### 3D Pocket Descriptor Features (CSV)

Each `*_descriptors_3d.csv` contains **109 features** per pocket.

**Geometric descriptors (RDKit3D)**:

* `Volume`
* `PMI1, PMI2, PMI3`
* `NPR1, NPR2`
* `Rgyr`
* `Asphericity`
* `SpherocityIndex`
* `Eccentricity`
* `InertialShapeFactor`

**Atom-type counts**:

* `CZ`, `CA`, `O`, `OD1`, `OG`, `N`, `NZ`, `DU`, etc.

**Burial/exposure features**:

* `CZ40`, `CZ50`, …, `CZ120`
* `T40`, `T50`, …, `T120`
* Similar depth bins for other atom types.

---

## Core Representation for QGNN

### Ligand (Drug) Features

* Source: ligand structures associated with PLOC / PLONC / PLA pockets

  * e.g., from `*.pdb`, `*.mol2`, or separate ligand files.
* Recommended ligand descriptors (via RDKit):

  * Molecular weight, LogP
  * H-bond donors / acceptors
  * Rotatable bonds
  * TPSA (topological polar surface area)
  * Simple 2D fingerprints (optional)

### Protein Pocket Features

* Source: `results/*_descriptors_3d.csv`
* Good starter subset (19 features):

  * `Volume, PMI1, PMI2, PMI3, NPR1, NPR2, Rgyr, Asphericity, SpherocityIndex, Eccentricity, InertialShapeFactor`
  * `CZ, CA, O, OD1, OG, N, NZ, DU`

### Interaction Labels

You can define supervised tasks like:

* **Binary binding**:

  * Positive: PLOC / PLONC / PLA pockets with ligands
  * Negative: HD pockets (no ligand).

* **Interaction type classification**:

  * Class 0: HD (no ligand / negative)
  * Class 1: PLOC (competitive)
  * Class 2: PLONC (non-competitive)
  * Class 3: PLA (allosteric)

---

## Graph Representation

We use a **bipartite graph**:

```text
Ligands (L1, L2, ..., Ln)    Protein Pockets (P1, P2, ..., Pm)
          │                           │
          ├────────────┬──────────────┤
          │            │              │
   Edge(L1,P3)   Edge(L2,P10)   Edge(L5,P7)
   [features]      [...]          [...]
```

* **Ligand nodes**:

  * Features: ligand descriptors (RDKit-based)
  * One node per unique ligand.

* **Protein pocket nodes**:

  * Features: 3D pocket descriptors (from CSV)
  * One node per pocket (PDB + chain + pocket ID).

* **Edges**:

  * A ligand binds a given pocket (from PLOC/PLONC/PLA).
  * Optional edge features: interaction type, interface distance, etc.

* **Labels**:

  * Binding / non-binding, or interaction class (competitive, allosteric, etc.).

---

## Quantum Encoding (Ligand–Protein Only)

The quantum circuit encodes **ligand–protein pairs**:

1. **Ligand qubits** (first half):

   * Encodes ligand latent features via `RY` rotations.

2. **Pocket qubits** (second half):

   * Encodes protein pocket latent features via `RY` rotations.

3. **Entanglement**:

   * CNOT or CRY gates between ligand and pocket qubits
   * Models interaction between their feature spaces.

4. **Variational layers**:

   * Parameterized rotations + entanglement repeated `L` times.

5. **Measurement**:

   * Expectation values of PauliZ on selected qubits → interaction embedding
   * Classical head → binding / class prediction.

---

## Installation

### Requirements

```bash
# Core dependencies
pip install torch numpy pandas

# Quantum computing
pip install pennylane

# Optional: GPU acceleration for quantum sims
pip install pennylane-lightning-gpu  # NVIDIA CUDA
# OR
pip install pennylane-lightning-mtl  # Apple Metal (M1/M2)

# Optional: Molecular processing
pip install rdkit-pypi biopython
```

---

## Example Usage (Drug–Protein Only)

```python
from drug_protein_quantum_gnn_pipeline import (
    DrugProteinDataProcessor,
    QuantumDrugProteinGNN,
    DrugProteinTrainer
)

# 1. Load and process ligand–protein data
processor = DrugProteinDataProcessor(data_dir="/media/priyanshu/SD/othercode/data")

# Load protein pocket descriptors (PLOC / PLONC / PLA / HD)
processor.load_protein_ligand_dataset(
    use_ploc=True,
    use_plonc=True,
    use_pla=True,
    use_hd=True
)

# Build bipartite ligand–pocket interaction graph
graph = processor.build_ligand_protein_graph()

# 2. Create model
ligand_dim = graph.get_ligand_feature_matrix().shape[1]
pocket_dim = graph.get_pocket_feature_matrix().shape[1]

model = QuantumDrugProteinGNN(
    ligand_dim=ligand_dim,
    pocket_dim=pocket_dim,
    num_qubits=8,     # 4 for ligand, 4 for pocket (example)
    num_qlayers=2,
    use_quantum=True  # set False for classical baseline
)

# 3. Train
trainer = DrugProteinTrainer(model, learning_rate=1e-3, device="cuda" or "cpu")
trainer.fit(graph, epochs=50, val_split=0.2, batch_size=256)

# 4. Save model
import torch
torch.save(model.state_dict(), "trained_drug_protein_qgnn.pt")
```

---

## Predicting on New Ligand–Protein Pairs

```python
# Load trained model
model = QuantumDrugProteinGNN(ligand_dim, pocket_dim, num_qubits=8)
model.load_state_dict(torch.load("trained_drug_protein_qgnn.pt"))
model.eval()

# Suppose you have:
#   ligand_vector: (ligand_dim,)
#   pocket_vector: (pocket_dim,)
ligand_features = torch.tensor(ligand_vector, dtype=torch.float32).unsqueeze(0)
pocket_features = torch.tensor(pocket_vector, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    binding_prob = model(ligand_features, pocket_features)
    print(f"Predicted binding probability: {binding_prob.item():.2%}")
```

---

## Pipeline Components

### 1. `DrugProteinDataProcessor`

**Purpose:** Load and preprocess drug–protein interaction data.

**Key Responsibilities:**

* Scan for `*_descriptors_3d.csv` under `data_dir`.
* Extract 3D pocket descriptors for protein pockets.
* Optionally compute ligand descriptors from ligand files (PDB/MOL2/SMILES).
* Build a bipartite graph:

  * Ligand nodes, pocket nodes, edges = binding interactions.
* Split into train/validation sets.

### 2. `QuantumDrugProteinGNN`

**Purpose:** QGNN model for ligand–protein interaction prediction.

**Architecture:**

* **Ligand encoder:**
  `Linear(ligand_dim → H) → ReLU → Linear(H → num_qubits)`
* **Pocket encoder:**
  `Linear(pocket_dim → H) → ReLU → Linear(H → num_qubits)`
* **Quantum layer:**

  * `num_qubits` for ligand + `num_qubits` for pocket
  * Angle encoding via `RY`
  * Entangling gates (e.g., CNOT or CRY) across ligand↔pocket wires
* **Output head:**
  `Linear(num_qubits → 32) → ReLU → Linear(32 → 1) → Sigmoid` (binary)
  or `Linear(32 → num_classes)` + softmax (multiclass).

### 3. `DrugProteinTrainer`

**Purpose:** Training and evaluation loop.

**Key Methods:**

* `train_epoch()`
* `evaluate()`
* `fit()` with train/val split

**Metrics:**

* Binary cross-entropy loss (for binding prediction)
* Accuracy, ROC-AUC, F1 (configurable)
* Multiclass loss/metrics if using multiple interaction types.

---

## Key Advantages of This Setup

1. **Biologically grounded**
   Uses real protein–ligand binding pockets from a curated dataset.

2. **Drug–Protein focused**
   No "patients" or EHR — just ligands and proteins.

3. **Quantum-friendly**
   Ligand and pocket embeddings are naturally mapped to two halves of a qubit register, with entanglement modeling interactions.

4. **Flexible tasks**
   Can handle binary binding, interaction type classification, or ranking.

---

## Future Extensions

* Add richer ligand descriptors (e.g., graph-based, fingerprints).
* Move from pocket-level to full protein graph representations.
* Integrate binding affinity (if available) as a regression target.
* Extend to multi-ligand, multi-pocket GNNs for polypharmacology.

---

## References

* Moine-Franel et al., *Scientific Data* (2024).
* Dataset DOI: [https://doi.org/10.5281/zenodo.10805580](https://doi.org/10.5281/zenodo.10805580)
* RCSB PDB: [https://www.rcsb.org/](https://www.rcsb.org/)
* RDKit: [https://www.rdkit.org/](https://www.rdkit.org/)
* PennyLane: [https://pennylane.ai/](https://pennylane.ai/)
