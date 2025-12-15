# Paper-Code Alignment Analysis

## Executive Summary

This document provides a detailed analysis of the alignment between the IEEE paper "A Hybrid Quantum Graph Neural Network Methodology for Drug-Protein Interaction Simulation" and the actual codebase implementation. The analysis identifies both areas of strong correspondence and significant discrepancies.

**Overall Assessment:** PARTIAL ALIGNMENT with several critical mismatches.

---

## 1. Dataset Claims vs. Implementation

### Paper Claims (Section 2.1)

**Dataset: CDPPILBP**
- 34,475 PDB structures
- 23,000 binding pockets
- 3,700 proteins across 500+ organisms
- 3,500 distinct ligands
- Total dataset size: 90 GB
- **244,260 ligand-pocket interactions** (122,130 positive, 122,130 negative)
- Data source: Zenodo (DOI: 10.5281/zenodo.10805580)
- MD5 checksum: a2ee28e5dea636e2892712609cd4c215

### Code Implementation (test.ipynb, data.py)

**Actual Usage:**
```python
# From test.ipynb, Cell 5
processor.load_data(max_samples=50)  # Only 50 protein files loaded
```

**Actual Statistics from Notebook Output:**
```
Found 50 protein descriptor files
Loaded 41 pockets, 364 ligands
Interactions: 364 positive, 364 negative
Total interactions: 728
```

**Dataset Split (test.ipynb, Cell 7):**
```
Train: 582 samples (80%)
Val:   73 samples (10%)
Test:  73 samples (10%)
```

### DISCREPANCY ANALYSIS

| Aspect | Paper Claim | Code Reality | Severity |
|--------|-------------|--------------|----------|
| Total interactions | 244,260 | 728 | CRITICAL |
| Positive samples | 122,130 | 364 | CRITICAL |
| Dataset used | Full CDPPILBP (90GB) | Tiny subset (50 files) | CRITICAL |
| Scale claim | "Large-scale dataset" | Toy dataset | CRITICAL |

**Verdict:** The paper claims to use a large-scale dataset of 244,260 interactions but the code only loads 728 interactions from 50 files. This is a **massive discrepancy** (99.7% reduction in scale).

---

## 2. Graph Construction

### Paper Claims (Section 2.5-2.8)

**Node Features:**
- Ligand atoms: Element type (C, N, O, P, halogens), formal charge, valence, ring indicators, hybridization (sp, sp2, sp3), H-bond donor/acceptor
- Pocket residues: 20 amino acids, secondary structure (alpha-helix, beta-sheet, loop)
- Pocket geometric features (100+ descriptors): Volume, PMI1-3, NPR1-2, Rgyr, Asphericity, Sphericity, Eccentricity, Inertial shape factor
- Atom type counts: CZ, CA, O, OD1, OG, N, NZ, DU

**Edge Features:**
- Ligand-ligand: Bond order (1, 2, 3, aromatic)
- Pocket-pocket: Residue proximity (C_alpha distance < 8 Angstrom)
- Ligand-pocket: Spatial proximity (heavy atom distance < 5 Angstrom)
- Edge weights: Gaussian RBF kernel with alpha = 0.5 A^-2

### Code Implementation (data.py)

**Ligand Node Features:**
```python
# Lines 76-82: Mol2GraphParser.get_atom_encoding()
ATOM_TYPES = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']
encoding = [1.0 if base_type == t else 0.0 for t in ATOM_TYPES]
# Returns 10-dimensional one-hot encodingx
```

**Pocket Features:**
```python
# Lines 41-52: PocketFeatures.to_vector()
geometric = [volume, pmi1, pmi2, pmi3, npr1, npr2, rgyr,
             asphericity, spherocity_index, eccentricity,
             inertial_shape_factor]  # 11 features
atom_types = ['CZ', 'CA', 'O', 'OD1', 'OG', 'N', 'NZ', 'DU']  # 8 features
# Total: 19 features
```

**Edge Construction:**
```python
# Lines 110-134: MOL2 bond parsing
bond_feat = 1.0  # Default single bond
if bond_type == '2': bond_feat = 2.0
elif bond_type == '3': bond_feat = 3.0
elif bond_type == 'ar': bond_feat = 1.5
elif bond_type == 'am': bond_feat = 1.5
# Edge attributes: (E, 1) - just bond order
```

### ALIGNMENT ANALYSIS

| Feature | Paper Description | Code Implementation | Status |
|---------|-------------------|---------------------|--------|
| Ligand atom types | C, N, O, P, halogens | 10-element one-hot (C, N, O, S, P, F, Cl, Br, I, H) | ALIGNED |
| Ligand hybridization | sp, sp2, sp3 | NOT IMPLEMENTED | MISSING |
| Ligand charge | Formal charge, valence | NOT IMPLEMENTED | MISSING |
| Ligand H-bond | Donor/acceptor flags | NOT IMPLEMENTED | MISSING |
| Pocket geometric | 100+ descriptors | 11 geometric features | PARTIAL |
| Pocket atom counts | CZ, CA, O, OD1, OG, N, NZ, DU | 8 atom type counts | ALIGNED |
| Pocket residues | 20 amino acids, secondary structure | NOT IMPLEMENTED | MISSING |
| Edge weights | Gaussian RBF (exp(-alpha*d^2)) | NOT IMPLEMENTED | MISSING |
| Ligand-pocket edges | Proximity < 5A | NOT IMPLEMENTED | MISSING |

**Verdict:** The paper describes a much richer feature set than implemented. The code uses:
- **10-dim ligand features** (just atom type) vs. paper's multi-faceted descriptor
- **19-dim pocket features** (11 geometric + 8 atom counts) vs. paper's 100+ descriptors
- **No edge weights** vs. paper's Gaussian RBF
- **No bipartite graph** (ligand-pocket edges missing)

---

## 3. Model Architecture

### Paper Claims (Section 3)

**Classical GNN (Equations 1-3):**
```
m_ij^(t) = phi([h_i^(t) || h_j^(t) || w_ij])
m_i^(t) = sum_{j in N(i)} m_ij^(t)
h_i^(t+1) = sigma(W_h h_i^(t) + W_m m_i^(t) + b_h)
```
- Message function phi: One-hidden-layer MLP (dim 64)
- Separate weight matrices W_h, W_m (both 64x64)
- Follows MPNN framework

**Shared Classical Pre-Embedding:**
```
h_i^(0) = sigma(W_0 x_i + b_0)
```
- Projects raw features to d_h = 64 dimensions

### Code Implementation (model.py)

**GCNLayer (Lines 6-67):**
```python
# Graph Convolution: H' = ReLU(D^-0.5 A D^-0.5 H W)
def forward(self, x, edge_index):
    # Compute normalized adjacency
    adj = torch.sparse_coo_tensor(full_edge_index, norm, (num_nodes, num_nodes))
    support = torch.sparse.mm(adj, x)
    out = self.linear(support)
    return out
```

**LigandGNN (Lines 69-98):**
```python
self.conv1 = GCNLayer(input_dim, hidden_dim)
self.conv2 = GCNLayer(hidden_dim, hidden_dim)
self.lin = nn.Linear(hidden_dim, output_dim)

def forward(self, x, edge_index, batch):
    x = F.relu(self.conv1(x, edge_index))
    x = F.relu(self.conv2(x, edge_index))
    # Global mean pooling
    pooled = ...  # Scatter-add aggregation
    x = self.lin(pooled)
    return x
```

**PocketMLP (Lines 100-113):**
```python
self.net = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)
```

### ALIGNMENT ANALYSIS

| Component | Paper Description | Code Implementation | Status |
|-----------|-------------------|---------------------|--------|
| Message passing | MLP-based with edge weights | GCN symmetric normalization | DIFFERENT |
| Message function phi | MLP([h_i, h_j, w_ij]) -> R^64 | Not present (GCN uses direct aggregation) | MISSING |
| Aggregation | Sum over neighbors | Sum (in GCN formula) | ALIGNED |
| Update function | W_h h_i + W_m m_i | Single linear layer in GCN | DIFFERENT |
| Edge weights | Used in message passing | Not used (adjacency normalization only) | MISSING |

**Verdict:** The code implements **standard GCN** (Kipf & Welling 2017), NOT the MPNN-style message passing described in the paper. This is a fundamental architectural difference.

---

## 4. Quantum Circuit Design

### Paper Claims (Section 4)

**Quantum Configuration:**
- Hilbert space: (C^2)^6 (6 qubits)
- Encoding: Angle encoding via R_y rotations
- Circuit depth: L = 2 layers
- Ansatz: Strongly Entangling Layers
- Entanglement: Ring topology CRY gates
- Entanglement edges: {(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)}
- Single-qubit rotations: R_z(alpha) R_y(beta) R_z(gamma) (3 angles per qubit)
- Measurement: Pauli-Z expectation on all qubits
- Total parameters: 48 (2 layers × (3×6 single-qubit + 6 CRY))
- Gradient: Parameter-shift rule

**Encoding (Section 4.2):**
```
|psi_i^(0)> = tensor_{k=1}^{d_h} R_y(h_ik^(0)) |0> ⊗ |0>^(n-d_h)
```
"First 6 dimensions of h_i^(0) in R^64 are selected"

### Code Implementation (model.py, quantum_parallel.py)

**QuantumInteractionLayer (model.py, Lines 115-189):**
```python
self.n_qubits = 6
self.n_layers = 2
self.device_name = 'lightning.qubit'

# Circuit definition (Lines 152-161)
@qml.qnode(self._dev, interface='torch', diff_method='best')
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
    return qml.expval(qml.PauliZ(0))  # Only qubit 0 measured

# Weight initialization (Line 164-165)
weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
self._q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
```

**ParallelQuantumInteractionLayer (quantum_parallel.py, Lines 48-52):**
```python
@qml.qnode(self._dev, interface='torch', diff_method='best')
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
    return qml.expval(qml.PauliZ(0))  # Only qubit 0 measured
```

**LigandPocketQGNN Forward Pass (model.py, Lines 231-251):**
```python
h_ligand = self.ligand_encoder(x, edge_index, batch)  # -> (B, 3)
h_pocket = self.pocket_encoder(pocket_vec)             # -> (B, 3)
combined = torch.cat([h_ligand, h_pocket], dim=1)      # -> (B, 6)
combined = torch.tanh(combined) * torch.pi             # Normalize to [0, 2pi]
out = self.interaction(combined)                       # -> (B, 1)
if self.use_quantum:
    out = (out + 1) / 2  # Map [-1, 1] to [0, 1]
```

### ALIGNMENT ANALYSIS

| Aspect | Paper Claim | Code Implementation | Status |
|--------|-------------|---------------------|--------|
| Number of qubits | 6 | 6 | ALIGNED |
| Circuit depth | 2 layers | 2 layers | ALIGNED |
| Encoding | Angle (R_y) | AngleEmbedding (R_y) | ALIGNED |
| Ansatz | Strongly Entangling | StronglyEntanglingLayers | ALIGNED |
| Entanglement topology | Ring CRY | PennyLane default (varies by version) | UNCERTAIN |
| Single-qubit rotations | R_z R_y R_z (3 per qubit) | Depends on PennyLane template | UNCERTAIN |
| Measurement | Pauli-Z on all qubits | Pauli-Z on qubit 0 ONLY | MISALIGNED |
| Total parameters | 48 | 2 × 6 × 3 = 36 (weights shape) | POTENTIAL MISMATCH |
| Input encoding | First 6 dims of h^(0) in R^64 | Direct 6-dim concatenation (3+3) | DIFFERENT |

**CRITICAL DISCREPANCY:**

**Paper (Section 4.5):**
> "h_iq^(L) = <psi_i^(L) | Z_q | psi_i^(L)>, q = 0, 1, ..., n-1"
> "The expectation values h_iq^(L) in [-1, 1] ... all 6 Pauli-Z expectations are computed"

**Code:**
```python
return qml.expval(qml.PauliZ(0))  # Only measures qubit 0
```

The code only measures qubit 0, but the paper claims to measure all 6 qubits.

**Verdict:** Core quantum design mostly aligns (6Q/2L, angle encoding, strongly entangling), but:
- **Measurement discrepancy:** Code measures 1 qubit, paper claims 6
- **Entanglement topology:** Paper specifies ring CRY, code uses PennyLane default (not explicitly verified)
- **Input preparation:** Code uses direct 6-dim vector (3 ligand + 3 pocket), paper suggests selecting first 6 from 64-dim embedding

---

## 5. Training Configuration

### Paper Claims (Section 5)

**Loss Function:**
```
L_BCE = -(1/N) sum[y_i log(y_hat_i) + (1-y_i) log(1-y_hat_i)]
```

**Optimizer:**
- Adam
- Classical GNN: learning rate = 1e-3
- QGNN: learning rate = 5e-4

**Data Split:**
- 80% training, 10% validation, 10% test
- Stratified by label

**Metrics:**
- Validation AUC (primary)
- Validation accuracy
- Validation precision
- Validation F1-score

**Early Stopping:**
- Based on validation AUC
- Patience not specified in paper

### Code Implementation (test.ipynb)

**Loss Function (Cell 9):**
```python
criterion = nn.BCELoss()  # Binary Cross-Entropy
```

**Optimizer (Cell 9):**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# LEARNING_RATE = 0.001 (Cell 3) - same for both models
```

**Data Split (Cell 7):**
```python
train_ints, temp_ints = train_test_split(interactions, test_size=0.2, random_state=SEED)
val_ints, test_ints = train_test_split(temp_ints, test_size=0.5, random_state=SEED)
# Results: 80/10/10 split
```

**Early Stopping (Cell 3, 9):**
```python
EARLY_STOPPING_PATIENCE = 15
# Stops after 15 epochs without AUC improvement
```

**Gradient Clipping (Cell 9):**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Learning Rate Scheduler (Cell 9):**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)
```

### ALIGNMENT ANALYSIS

| Aspect | Paper Claim | Code Implementation | Status |
|--------|-------------|---------------------|--------|
| Loss function | BCE | nn.BCELoss() | ALIGNED |
| Optimizer | Adam | torch.optim.Adam | ALIGNED |
| Classical LR | 1e-3 | 1e-3 | ALIGNED |
| QGNN LR | 5e-4 | 1e-3 | MISALIGNED |
| Data split | 80/10/10 | 80/10/10 | ALIGNED |
| Metrics | AUC, Acc, Prec, F1 | AUC, Acc, Prec, Recall, F1 | ALIGNED+ |
| Early stopping | Yes (unspecified) | 15 epochs patience | ALIGNED |
| Gradient clipping | Not mentioned | max_norm=1.0 | CODE ADDITION |
| LR scheduling | Not mentioned | ReduceLROnPlateau | CODE ADDITION |

**Verdict:** Training setup mostly aligns, except:
- **Learning rate:** Paper claims different LR for QGNN (5e-4), but code uses same LR (1e-3) for both
- **Code implements additional features** not mentioned in paper (gradient clipping, LR scheduling)

---

## 6. Results and Performance

### Paper Claims (Table 1, Section 6)

**Validation Performance (Configuration B):**

| Model | AUC | Accuracy | Precision | F1-Score |
|-------|-----|----------|-----------|----------|
| Classical GNN | 0.8033 | 0.7222 | 0.7555 | 0.7035 |
| QGNN (6Q/2L) | 0.6854 | 0.6539 | 0.6868 | 0.6218 |

**Paper Claims:**
- "Classical model achieves higher final performance"
- "QGNN exhibits meaningful early epoch learning behavior"
- "QGNN converges rapidly within 10 epochs"
- "Classical continues to improve with additional training"

### Code Results (test.ipynb outputs)

**Validation Performance (from notebook):**

| Model | AUC | Accuracy | Precision | Recall | F1 |
|-------|-----|----------|-----------|--------|-----|
| Quantum | 0.6805 | 0.6438* | 0.4706 | 0.7742 | 0.5854 |
| Classical | 0.6071 | 0.6027* | 0.4030 | 0.8710 | 0.5510 |

*Accuracy values from comparison table show "None" - these are from test set

**Test Set Performance (from notebook):**

| Model | Loss | Accuracy | AUC | Precision | Recall | F1 |
|-------|------|----------|-----|-----------|--------|-----|
| Quantum | 1.5508 | 0.6438 | 0.5818 | 0.6538 | 0.8095 | 0.7234 |
| Classical | 0.6865 | 0.6027 | 0.6544 | 0.5970 | 0.9524 | 0.7339 |

**Training Behavior:**
- Classical: Trained for 28 epochs, early stopped
- Quantum: Shows checkpoint loading with previous training

### DISCREPANCY ANALYSIS

| Metric | Paper (Classical) | Code (Classical) | Difference |
|--------|-------------------|------------------|------------|
| Val AUC | 0.8033 | 0.6071 | -0.1962 (-24.4%) |
| Val Acc | 0.7222 | 0.6027* | -0.1195 (-16.5%) |
| Val Prec | 0.7555 | 0.4030 | -0.3525 (-46.7%) |
| Val F1 | 0.7035 | 0.5510 | -0.1525 (-21.7%) |

| Metric | Paper (QGNN) | Code (QGNN) | Difference |
|--------|--------------|-------------|------------|
| Val AUC | 0.6854 | 0.6805 | -0.0049 (-0.7%) |
| Val Acc | 0.6539 | 0.6438* | -0.0101 (-1.5%) |
| Val Prec | 0.6868 | 0.4706 | -0.2162 (-31.5%) |
| Val F1 | 0.6218 | 0.5854 | -0.0364 (-5.9%) |

**CRITICAL FINDING:**
In the paper, Classical > Quantum for all metrics.
In the code, **Classical < Quantum** for AUC (0.6071 vs 0.6805) on validation, but Classical > Quantum on test set.

**Verdict:**
- **Results do NOT match** the paper's reported values
- **Performance ranking is inconsistent**: Paper shows classical superiority, code shows mixed results
- **Magnitude differences are large** (up to 46.7% for precision)
- This strongly suggests the paper results came from a **different experimental run** with different data/configuration

---

## 7. Hardware Optimization

### Paper Claims

**No explicit hardware optimization discussion** in the paper.

### Code Implementation (hardware_optimizer.py)

**Comprehensive auto-detection:**
- Detects CPU vendor (Apple, Intel, AMD, ARM, Qualcomm)
- Detects GPU (CUDA, MPS)
- Optimizes batch size, num_workers, prefetch_factor based on hardware
- Selects quantum backend (lightning.gpu for CUDA, lightning.qubit for CPU/MPS)

**Example from test.ipynb output:**
```
HARDWARE DETECTED
Platform:     Darwin
CPU:          Apple M4 Pro
Device:       Apple Metal Performance Shaders
Batch Size:   1024
Workers:      10
Quantum Dev:  lightning.qubit
```

**Verdict:** Hardware optimization is a significant **code feature not mentioned in the paper**.

---

## 8. Computational Complexity

### Paper Claims (Section 6.4)

> "Training a QGNN is substantially more expensive per epoch than training its classical counterpart... every epoch runtimes hours longer"
> "Simulation using PennyLane backend"
> "Cannot be efficiently parallelized across samples"

### Code Implementation

**Parallel Quantum Layer (quantum_parallel.py):**
```python
class ParallelQuantumInteractionLayer:
    # Uses ThreadPoolExecutor for parallel evaluation
    n_workers = os.cpu_count()
    self._executor = ThreadPoolExecutor(max_workers=n_workers)

    # Parallel execution (Lines 86-90)
    futures = [self._executor.submit(eval_single, i) for i in range(batch_size)]
    results = [future.result() for future in futures]
```

**Documentation claims:**
> "Parallel quantum layer for multi-core processing"
> "12 worker threads for parallel evaluation"
> "Speedup: ~Nx where N = CPU cores"

**Verdict:** The code **directly contradicts** the paper's claim about parallelization impossibility. The implementation includes a custom parallel quantum layer not mentioned in the paper.

---

## 9. Figures and Visualizations

### Paper Figures

- **Figure 1 (biophysical_graph):** Protein-to-graph conversion with bipartite structure
- **Figure 2 (hybrid_arch):** Hybrid architecture flowchart
- **Figure 3 (quantum_circuit):** 6Q/2L circuit diagram with ring CRY
- **Figure 4 (training_curves):** "figs/training_curves_full.png"
- **Figure 5 (epoch_aligned):** "figs/epoch_aligned_27.png"

### Code Outputs (test.ipynb)

**Cell 17:** Generates comparison plots
```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics = [('train_loss', 'Train Loss'), ('val_loss', 'Val Loss'),
           ('train_acc', 'Train Acc'), ('val_acc', 'Val Acc'),
           ('val_auc', 'Val AUC'), ('val_f1', 'Val F1')]
plt.savefig(os.path.join(SAVE_DIR, 'comparison_plots.png'), dpi=300)
```

**Verdict:** Code generates plots, but no "figs/" directory with paper figures exists in the repository. The paper figures appear to be from a **separate experimental run**.

---

## 10. Reproducibility Assessment

### What is Needed to Reproduce Paper Results

1. **Full CDPPILBP dataset** (244,260 interactions)
2. **Bipartite graph construction** (ligand-pocket edges)
3. **Rich feature set** (100+ pocket descriptors, ligand chemistry)
4. **Gaussian RBF edge weights**
5. **MPNN-style message passing** (not GCN)
6. **Different learning rates** (5e-4 for QGNN)
7. **Multi-qubit measurement** (all 6 qubits)
8. **Longer training runs** to reach reported performance

### What the Code Actually Does

1. **Toy dataset** (728 interactions from 50 files)
2. **Ligand-only graph** (no bipartite structure)
3. **Minimal features** (10-dim ligand, 19-dim pocket)
4. **No edge weights** (just normalization)
5. **GCN message passing** (symmetric normalization)
6. **Same learning rate** (1e-3 for both)
7. **Single-qubit measurement** (only qubit 0)
8. **Short runs** (28 epochs, early stopping)

### Reproducibility Verdict

**The current code CANNOT reproduce the paper results.** Major modifications would be required:

- [ ] Load full CDPPILBP dataset (244,260 interactions)
- [ ] Implement bipartite graph construction
- [ ] Add rich feature extraction (100+ descriptors)
- [ ] Implement Gaussian RBF edge weighting
- [ ] Replace GCN with MPNN message passing
- [ ] Modify quantum circuit to measure all 6 qubits
- [ ] Update pooling to aggregate 6 measurements per node
- [ ] Adjust hyperparameters (QGNN LR = 5e-4)
- [ ] Train for sufficient epochs to converge

---

## Summary Table: Paper vs. Code Alignment

| Component | Paper Description | Code Implementation | Alignment Score |
|-----------|-------------------|---------------------|-----------------|
| **Dataset** | 244,260 interactions, full CDPPILBP | 728 interactions, 50 files | 0/10 |
| **Graph Structure** | Bipartite (ligand-pocket edges) | Ligand-only | 3/10 |
| **Node Features** | 100+ pocket, rich ligand | 19 pocket, 10 ligand | 4/10 |
| **Edge Features** | Gaussian RBF weights | No weights | 1/10 |
| **Classical GNN** | MPNN message passing | GCN convolution | 5/10 |
| **Quantum Circuit** | 6Q/2L, angle, strongly ent. | 6Q/2L, angle, strongly ent. | 8/10 |
| **Quantum Measurement** | All 6 qubits (Pauli-Z) | Only qubit 0 (Pauli-Z) | 3/10 |
| **Training Loss** | BCE | BCE | 10/10 |
| **Optimizer** | Adam (1e-3, 5e-4) | Adam (1e-3, 1e-3) | 7/10 |
| **Data Split** | 80/10/10 | 80/10/10 | 10/10 |
| **Performance** | Classical AUC 0.80 | Classical AUC 0.61 | 2/10 |
| **Parallelization** | "Cannot parallelize" | Parallel quantum layer | -5/10 |
| **Hardware Opt** | Not mentioned | Comprehensive auto-detect | N/A |

**Overall Alignment Score: 48/120 (40%)**

---

## Recommendations

### For Academic Integrity

1. **Update paper abstract** to state: "This work presents a methodology and preliminary results on a subset of the CDPPILBP dataset"
2. **Clarify dataset size** in Section 2: "We use 728 interactions from 50 protein structures as a proof-of-concept"
3. **Revise results section** to match actual code outputs
4. **Add limitations section** acknowledging:
   - Small-scale validation (not large-scale)
   - Simplified graph construction (no bipartite structure)
   - Minimal feature set (10+19 vs. 100+)
   - Code-paper discrepancies

### For Code Improvement

1. **Implement full dataset loading** (remove max_samples=50 hardcoding)
2. **Add bipartite graph construction** (ligand-pocket edges)
3. **Expand feature extraction** to match paper description
4. **Implement MPNN-style message passing** or update paper to describe GCN
5. **Modify quantum circuit measurement** to all 6 qubits
6. **Add configuration file** to control QGNN vs Classical LR separately
7. **Include paper figures** in repository (figs/ directory)
8. **Add reproducibility script** that matches paper experiments

### For Future Work

1. **Run full-scale experiments** on complete CDPPILBP dataset
2. **Validate performance claims** with proper statistical analysis
3. **Compare parallelization approaches** (sequential vs. threaded quantum)
4. **Benchmark hardware requirements** (memory, time) at scale
5. **Publish updated results** with corrected figures and tables

---

## Conclusion

The paper and code represent **different stages** of the research project:

- **Paper:** Describes an ambitious, comprehensive methodology with large-scale validation
- **Code:** Implements a proof-of-concept on a toy dataset with simplified architecture

The discrepancies are **systematic and significant**, affecting:
- Dataset scale (99.7% smaller)
- Graph structure (missing bipartite connections)
- Feature richness (90% fewer descriptors)
- Model architecture (GCN vs. MPNN)
- Measurement strategy (1 qubit vs. 6 qubits)
- Performance outcomes (classical AUC: 0.80 vs. 0.61)

**This is NOT a case of minor implementation details differing from theory.** The code appears to be an earlier prototype that was later extended for the paper experiments, but the paper results were not generated by the current codebase.

**Recommendation:** Either update the paper to match the code, or update the code to match the paper. Currently, they represent incompatible versions of the same project.
