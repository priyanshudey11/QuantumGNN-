# QuantumGNN Project Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Module Documentation](#module-documentation)
4. [Data Pipeline](#data-pipeline)
5. [Model Components](#model-components)
6. [Hardware Optimization](#hardware-optimization)
7. [Training Workflow](#training-workflow)
8. [Usage Guide](#usage-guide)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

### Purpose

The QuantumGNN project implements a hybrid quantum-classical Graph Neural Network for predicting ligand-pocket binding interactions. The system compares quantum circuit-based interaction layers against classical neural networks for drug discovery applications.

### Key Features

- Hybrid quantum-classical architecture combining GNNs and quantum circuits
- Automatic hardware detection and optimization for CUDA, MPS (Apple Silicon), and CPU
- Parallel quantum circuit evaluation for improved training speed
- Comprehensive data processing pipeline for molecular structures
- Checkpoint-based training with patience-aware resumption

### Scientific Domain

This framework addresses the computational challenge of predicting whether a small molecule (ligand/drug) will bind to a protein pocket. The problem is formulated as binary classification:
- Input: Ligand graph structure + Pocket geometric features
- Output: Probability of binding interaction (0-1)

---

## Architecture

### System Components

```
QuantumGNN Pipeline
├── Data Layer (data.py)
│   ├── PocketFeatures: 3D geometric descriptors
│   ├── LigandGraph: Molecular graph representation
│   └── LigandPocketDataProcessor: Data loading and preprocessing
│
├── Model Layer (model.py)
│   ├── LigandGNN: Graph Convolutional Network for ligands
│   ├── PocketMLP: Multi-layer perceptron for pockets
│   ├── QuantumInteractionLayer: Standard quantum circuit
│   ├── ParallelQuantumInteractionLayer: Multi-threaded quantum evaluation
│   └── LigandPocketQGNN: Complete hybrid model
│
├── Hardware Layer (hardware_optimizer.py)
│   ├── detect_hardware(): Auto-detect CPU/GPU capabilities
│   ├── optimize_for_hardware(): Configure batch size, workers
│   └── setup_environment(): Initialize optimized environment
│
└── Training Layer (test.ipynb)
    ├── Training loop with early stopping
    ├── Checkpoint management with patience tracking
    └── Performance evaluation and visualization
```

### Data Flow

```
Raw Data (MOL2 + CSV)
    ↓
LigandPocketDataProcessor
    ↓
PyTorch Dataset + DataLoader
    ↓
LigandGNN (Graph) ⊕ PocketMLP (Vector)
    ↓
Concatenated Latent Vector (6 qubits)
    ↓
Quantum/Classical Interaction Layer
    ↓
Binding Probability (Sigmoid Output)
```

---

## Module Documentation

### data.py

#### PocketFeatures

Data class representing protein binding pocket characteristics.

**Attributes:**
- `pocket_id` (str): Unique identifier
- Geometric features (11 dimensions):
  - `volume`: Pocket volume in cubic Angstroms
  - `pmi1, pmi2, pmi3`: Principal moments of inertia
  - `npr1, npr2`: Normalized principal ratios
  - `rgyr`: Radius of gyration
  - `asphericity`: Deviation from spherical shape
  - `spherocity_index`: Measure of sphericity
  - `eccentricity`: Shape eccentricity
  - `inertial_shape_factor`: Inertial shape descriptor
- Atom type counts (8 dimensions): CZ, CA, O, OD1, OG, N, NZ, DU

**Methods:**
- `to_vector() -> np.ndarray`: Convert to 19-dimensional feature vector

**Example:**
```python
pocket = PocketFeatures(pocket_id="1abc")
pocket.volume = 1200.5
vector = pocket.to_vector()  # Shape: (19,)
```

#### LigandGraph

Data class for molecular graph representation.

**Attributes:**
- `ligand_id` (str): Unique identifier
- `atom_features` (np.ndarray): Node features, shape (N_atoms, 10)
  - One-hot encoding of atom types: C, N, O, S, P, F, Cl, Br, I, H
- `edge_index` (np.ndarray): Graph connectivity, shape (2, N_edges)
  - COO format: [source_nodes, target_nodes]
- `edge_attributes` (np.ndarray): Edge features, shape (N_edges, 1)
  - Bond type: 1.0 (single), 2.0 (double), 3.0 (triple), 1.5 (aromatic/amide)

**Example:**
```python
ligand = Mol2GraphParser.parse("molecule.mol2")
print(ligand)  # LigandGraph(id=molecule, atoms=42, edges=88)
```

#### Mol2GraphParser

Static parser for MOL2 chemical file format.

**Method:**
- `parse(filepath: str) -> Optional[LigandGraph]`
  - Parses MOL2 file sections (ATOM, BOND)
  - Returns LigandGraph or None on error

**File Format Expected:**
```
@<TRIPOS>ATOM
1 C1 2.5 1.3 0.8 C.3 ...
@<TRIPOS>BOND
1 1 2 1
```

#### LigandPocketDataProcessor

Main data loading and processing class.

**Initialization:**
```python
processor = LigandPocketDataProcessor(
    data_dir="/path/to/data",
    seed=42
)
```

**Methods:**

1. `load_data(max_samples: Optional[int] = None)`
   - Searches for CSV descriptor files recursively
   - Parses pocket features from CSV
   - Loads associated MOL2 ligand files
   - Generates negative samples (random non-binding pairs)
   - Creates balanced dataset (1:1 positive:negative ratio)

2. `get_dataset() -> List[LigandPocketInteraction]`
   - Returns list of all interactions

**Data Directory Structure:**
```
data/
└── protein_name/
    └── results/
        └── pocket_id/
            ├── pocket_id_descriptors_3d.csv
            ├── ligand1.mol2
            └── ligand2.mol2
```

**Negative Sampling Strategy:**
- Vectorized batch generation for speed
- Ensures no overlap with positive pairs
- Maintains equal positive/negative balance

#### LigandPocketDataset

PyTorch Dataset wrapper.

**Usage:**
```python
dataset = LigandPocketDataset(processor, interactions)
x, edge_idx, pocket_vec, label = dataset[0]
```

**Returns:**
- `x`: Atom features tensor
- `edge_idx`: Edge index tensor
- `pocket_vec`: Pocket feature vector
- `label`: Binary label (0.0 or 1.0)

#### collate_fn

Custom batch collation function for graph data.

**Functionality:**
- Concatenates graphs into a single large graph
- Shifts edge indices to maintain connectivity
- Creates batch index vector for graph pooling
- Stacks pocket features and labels

**Performance Optimizations:**
- Single-pass batch index creation
- Vectorized concatenation
- Pre-allocation of tensors

---

### model.py

#### GCNLayer

Graph Convolutional Network layer with symmetric normalization.

**Architecture:**
```
H' = ReLU(D^(-0.5) * A * D^(-0.5) * H * W)
```

Where:
- A: Adjacency matrix with self-loops
- D: Degree matrix
- H: Node features
- W: Learnable weight matrix

**Special Handling:**
- MPS device compatibility: Falls back to CPU for sparse operations
- Self-loops automatically added
- Normalized adjacency for stable gradients

**Forward Pass:**
```python
layer = GCNLayer(in_features=10, out_features=64)
output = layer(x, edge_index)
```

#### LigandGNN

Two-layer GCN with global mean pooling.

**Architecture:**
```
Input (N_atoms, 10)
    ↓ GCNLayer + ReLU
Hidden (N_atoms, hidden_dim)
    ↓ GCNLayer + ReLU
Hidden (N_atoms, hidden_dim)
    ↓ Global Mean Pooling
Output (batch_size, output_dim)
    ↓ Linear Projection
Latent Vector (batch_size, output_dim)
```

**Pooling:**
- Scatter-add aggregation by batch index
- Division by node counts for mean pooling
- Handles variable-size graphs in batch

#### PocketMLP

Three-layer fully connected network for pocket encoding.

**Architecture:**
```
Input (19,)
    ↓ Linear + ReLU
Hidden (hidden_dim,)
    ↓ Linear + ReLU
Hidden (hidden_dim,)
    ↓ Linear
Output (output_dim,)
```

**No Activation on Output:**
- Output is latent representation, not classification
- Allows negative values for quantum encoding

#### QuantumInteractionLayer

Standard quantum circuit layer for single-sample evaluation.

**Lazy Initialization:**
- Circuit created on first forward pass
- Enables multiprocessing (DataLoader workers)
- Registers trainable weights via TorchLayer

**Quantum Circuit Structure:**
```
1. AngleEmbedding: Encode classical data into rotation angles
   |0⟩ → RY(input[0]) → RY(input[1]) → ...

2. StronglyEntanglingLayers: Variational ansatz (n_layers deep)
   Each layer: RX(θ), RY(φ), RZ(ω) on each qubit + CNOT entanglers

3. Measurement: Expectation value of Z on qubit 0
   Output ∈ [-1, 1]
```

**Parameters:**
- Weight shape: (n_layers, n_qubits, 3)
- Total params: n_layers × n_qubits × 3

**Environment Setup:**
- Sets `OMP_NUM_THREADS` to CPU count
- Enables OpenMP parallelism in PennyLane

#### ParallelQuantumInteractionLayer (quantum_parallel.py)

Multi-threaded quantum circuit evaluation for batch processing.

**Key Differences from Standard Layer:**
- Uses ThreadPoolExecutor for parallel sample evaluation
- Processes batch samples concurrently
- Better performance on multi-core CPUs

**Architecture:**
- Same circuit structure as QuantumInteractionLayer
- Thread pool size = CPU core count
- Small batches (≤2) use sequential evaluation (avoid overhead)

**Thread Safety:**
- Moves tensors to CPU before threading (avoids GPU context issues)
- Each thread evaluates one sample independently
- Results collected and stacked

**Performance Characteristics:**
- Speedup scales with CPU cores
- Overhead for small batches
- Ideal for batch_size ≥ 16

**Device Handling:**
```python
x_cpu = x.cpu()  # Move to CPU for thread safety
weights_cpu = self._weights.cpu()

# Parallel evaluation
results = parallel_map(qnode, x_cpu, weights_cpu)

# Move back to original device
outputs = outputs.to(x.device)
```

#### LigandPocketQGNN

Complete hybrid quantum-classical model.

**Initialization Parameters:**
- `ligand_in_dim`: Atom feature dimension (default: 10)
- `pocket_in_dim`: Pocket feature dimension (default: 19)
- `hidden_dim`: Hidden layer size (default: 64)
- `latent_dim`: Latent vector size (default: 6)
- `n_qubits`: Number of qubits (default: 6)
- `n_qlayers`: Quantum circuit depth (default: 2)
- `use_quantum`: Toggle quantum vs classical interaction (default: True)
- `quantum_device`: PennyLane device name (default: 'lightning.qubit')
- `use_parallel`: Enable parallel quantum layer (default: True)

**Forward Pass:**
```python
h_ligand = LigandGNN(x, edge_index, batch)      # Shape: (B, 3)
h_pocket = PocketMLP(pocket_vec)                 # Shape: (B, 3)
combined = concat([h_ligand, h_pocket])          # Shape: (B, 6)
combined = tanh(combined) * π                    # Normalize to [0, 2π]
output = QuantumLayer(combined)                  # Shape: (B, 1)
output = (output + 1) / 2                        # Map [-1,1] to [0,1]
```

**Classical Fallback:**
If `use_quantum=False`, replaces quantum layer with:
```python
nn.Sequential(
    nn.Linear(n_qubits, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1),
    nn.Sigmoid()
)
```

**Model Size:**
- Quantum model: ~10,694 parameters
- Classical model: ~11,207 parameters

---

### hardware_optimizer.py

#### detect_hardware()

Comprehensive hardware detection across platforms.

**Detection Strategy:**

**macOS:**
- Uses `sysctl` for CPU brand and memory
- Detects Apple Silicon (M1/M2/M3/M4)
- Identifies Intel Macs

**Linux:**
- Parses `/proc/cpuinfo` for CPU information
- Reads `/proc/meminfo` for memory
- Detects AMD (EPYC, Threadripper, Ryzen), Intel (Xeon, i9/i7/i5), ARM (Neoverse), Qualcomm (Snapdragon)

**Windows:**
- Uses `wmi` library for hardware queries
- Fallback to basic detection if WMI unavailable

**GPU Detection:**
- CUDA: `torch.cuda` API for NVIDIA GPUs
  - Extracts compute capability (SM version)
  - Classifies architecture: Hopper (H100), Ampere (A100, RTX 40), Ada (RTX 40), Turing
- MPS: `torch.backends.mps` for Apple Silicon GPUs

**Return Value:**
```python
{
    'platform': 'Darwin',
    'cpu_count': 12,
    'cpu_brand': 'Apple M4 Pro',
    'cpu_vendor': 'apple',
    'device': 'mps',
    'device_name': 'Apple Metal Performance Shaders',
    'memory_gb': 24.0,
    'compute_capability': 'apple_m4_gpu'
}
```

#### optimize_for_hardware(hw_info)

Determines optimal training hyperparameters based on hardware.

**Configuration Parameters:**
- `batch_size`: Samples per training batch
- `num_workers`: DataLoader worker processes
- `prefetch_factor`: Batches to prefetch per worker
- `pin_memory`: Enable pinned memory for GPU transfer
- `quantum_device`: PennyLane quantum backend
- `persistent_workers`: Keep workers alive between epochs

**Optimization Rules:**

**NVIDIA CUDA:**
| GPU Tier | Memory | Batch Size | Workers | Quantum Device |
|----------|--------|------------|---------|----------------|
| Datacenter (A100/H100) | 40-80 GB | 4096 | 12 | lightning.gpu |
| High-end (RTX 4090) | 20+ GB | 2048 | 8 | lightning.gpu |
| Upper mid (RTX 4070 Ti) | 12+ GB | 1024 | 6 | lightning.gpu |
| Mid-range (RTX 4060 Ti) | 8+ GB | 512 | 4 | lightning.gpu |
| Entry-level | <8 GB | 256 | 4 | lightning.gpu |

**Apple MPS:**
| Chip | Batch Size | Workers | Notes |
|------|------------|---------|-------|
| M4 | 1536 | 10 | Latest, best perf |
| M3 | 1024 | 8 | High performance |
| M2 | 768 | 6 | Good performance |
| M1 | 512 | 4 | Still capable |

**CPU-Only:**
| CPU Type | Cores | Batch Size | Workers | Notes |
|----------|-------|------------|---------|-------|
| AMD EPYC/Threadripper | 32+ | 3072 | 16 | Server-grade |
| AMD Ryzen 9 | 16+ | 2048 | 12 | High-end desktop |
| Intel Xeon | 16+ | 2048 | 12 | Server |
| Intel i9 | 16+ | 1536 | 10 | High-end |
| ARM Server (Graviton) | 64+ | 2048 | 16 | Cloud instances |

**Memory Constraints:**
- <8 GB: Cap batch size at 256
- <16 GB: Cap at 512
- <32 GB: Cap at 1024

**Return Example:**
```python
{
    'batch_size': 1024,
    'num_workers': 10,
    'prefetch_factor': 5,
    'pin_memory': True,
    'quantum_device': 'lightning.qubit',
    'persistent_workers': True
}
```

#### setup_environment()

Convenience function combining detection and optimization.

**Usage:**
```python
hw_info, config = setup_environment()
# Prints formatted hardware info
# Returns both detection results and optimized config
```

---

## Data Pipeline

### Input Data Requirements

**Protein Pocket Descriptors (CSV):**
- Filename pattern: `*_descriptors_3d.csv`
- Required columns: Volume, PMI1, PMI2, PMI3, NPR1, NPR2, RGYR, Asphericity, Spherocity_Index, Eccentricity, Inertial_Shape_Factor, atom type counts (CZ, CA, O, OD1, OG, N, NZ, DU)

**Ligand Structures (MOL2):**
- Must be in same directory as descriptor CSV
- Standard MOL2 format with ATOM and BOND sections
- Supports bond types: single (1), double (2), triple (3), aromatic (ar), amide (am)

### Data Processing Pipeline

1. **Discovery Phase:**
   - Recursive search for `*_descriptors_3d.csv` files
   - Extract pocket ID from filename
   - Locate co-located MOL2 files

2. **Pocket Feature Extraction:**
   - Parse CSV row into PocketFeatures dataclass
   - Normalize geometric features
   - Count atom types

3. **Ligand Graph Construction:**
   - Parse MOL2 file structure
   - Extract atom coordinates and types
   - Build edge list from bond section
   - Create bidirectional edges (undirected graph)
   - One-hot encode atom types

4. **Positive Interaction Creation:**
   - Link each ligand to its co-located pocket
   - Label as binding (1.0)

5. **Negative Sampling:**
   - Generate random ligand-pocket pairs
   - Filter out existing positive pairs
   - Match count to positive samples
   - Label as non-binding (0.0)

6. **Dataset Splitting:**
   - Train: 80%
   - Validation: 10%
   - Test: 10%
   - Stratified by label (maintains balance)

### Batch Collation

**Challenge:** Graphs have variable sizes (different atom counts).

**Solution:**
1. Concatenate all graphs into single large graph
2. Track batch assignment for each node
3. Shift edge indices to maintain connectivity
4. Stack fixed-size pocket vectors

**Example:**
```
Graph 1: 10 nodes, edges [[0,1], [1,2]]
Graph 2: 5 nodes, edges [[0,1]]

Batched:
Nodes: 15 total
Edges: [[0,1], [1,2], [10,11]]  # Graph 2 edges shifted by +10
Batch: [0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1]
```

---

## Model Components

### Graph Neural Network Theory

**Message Passing Framework:**
1. Message: Aggregate neighbor features
2. Update: Transform node features with aggregated messages
3. Readout: Pool node features to graph-level representation

**GCN Normalization:**
- Prevents feature explosion in deep networks
- Balances contribution from high/low-degree nodes
- Symmetric normalization ensures smoothing

### Quantum Circuit Design

**Angle Embedding:**
- Maps classical data to quantum states
- Encodes 6D vector as rotation angles
- Each qubit initialized via RY gate

**Strongly Entangling Layers:**
- Universal quantum circuit ansatz
- Each layer applies:
  - Single-qubit rotations: RX(θ), RY(φ), RZ(ω)
  - Entangling gates: CNOT between adjacent qubits
- Depth controlled by `n_qlayers` parameter

**Measurement:**
- Pauli-Z expectation on first qubit
- Output range: [-1, 1]
- Post-processing: Linear map to [0, 1] for probability

**Theoretical Advantages:**
- Exponential parameter efficiency (hypothesis)
- Non-linear feature interactions
- Potential quantum advantage for specific data distributions

**Practical Considerations:**
- Simulated quantum circuits (classical overhead)
- Gradient computation via parameter-shift rule
- Limited qubit count (6) due to simulation constraints

### Classical Baseline

**Architecture:**
- Two-layer MLP with ReLU activation
- Same input dimension (6) as quantum layer
- Sigmoid output for probability
- More parameters than quantum layer

**Purpose:**
- Ablation study: isolate quantum contribution
- Fair comparison (similar capacity)
- Validate quantum performance claims

---

## Hardware Optimization

### Batch Size Selection

**Trade-offs:**
- Larger batches: Better GPU utilization, less noise in gradients
- Smaller batches: More frequent updates, better generalization
- Memory constraint: Batch size × model size ≤ available memory

**Adaptive Strategy:**
- Scale with available memory
- Adjust for GPU/CPU architecture
- Balance with DataLoader workers

### DataLoader Workers

**Purpose:**
- Parallel data loading and preprocessing
- Overlap data loading with GPU computation
- Prevent I/O bottleneck

**Optimal Count:**
- Too few: GPU starvation (waiting for data)
- Too many: CPU thrashing, memory overhead
- Sweet spot: 4-12 workers depending on CPU cores

**Persistent Workers:**
- Keep worker processes alive between epochs
- Avoid spawn overhead
- Trade memory for speed

### Prefetching

**Mechanism:**
- Each worker pre-loads N batches ahead
- GPU consumes batches while workers prepare next
- Pipeline parallelism

**Configuration:**
- Prefetch factor × num_workers = total prefetched batches
- Higher values: More memory, better throughput
- Lower values: Less memory, potential stalls

### Device-Specific Optimizations

**CUDA:**
- Pinned memory for async transfer
- cuDNN benchmark mode for conv operations
- lightning.gpu quantum backend (GPU-accelerated simulation)

**MPS (Apple Silicon):**
- Unified memory architecture (CPU/GPU share RAM)
- Pinned memory still beneficial
- Sparse operations fallback to CPU

**CPU:**
- Larger batch sizes (no GPU memory limit)
- More workers (exploit multi-core parallelism)
- OpenMP threading for quantum simulation

---

## Training Workflow

### Checkpoint System

**Checkpoint Contents:**
```python
{
    'model_state_dict': OrderedDict(...),  # Model weights
    'best_auc': 0.6805,                    # Best validation AUC
    'epoch': 42,                           # Epoch when saved
    'patience_counter': 3                  # Current patience count
}
```

**History File (JSON):**
```json
{
    "train_loss": [0.69, 0.68, ...],
    "train_acc": [0.50, 0.52, ...],
    "val_loss": [0.70, 0.69, ...],
    "val_acc": [0.49, 0.51, ...],
    "val_auc": [0.50, 0.58, ...],
    "val_precision": [0.48, 0.50, ...],
    "val_recall": [0.52, 0.55, ...],
    "val_f1": [0.50, 0.52, ...]
}
```

### Training Loop

**Epoch Structure:**
1. Training phase:
   - Forward pass through model
   - Compute BCE loss
   - Backward pass (gradient computation)
   - Gradient clipping (max_norm=1.0)
   - Optimizer step (Adam)

2. Validation phase:
   - No gradient computation
   - Compute metrics: loss, accuracy, AUC, precision, recall, F1
   - Learning rate scheduling (ReduceLROnPlateau)

3. Checkpoint decision:
   - If AUC improves: Save checkpoint, reset patience
   - If AUC stagnates: Increment patience
   - If patience exhausted: Early stop

### Early Stopping with Patience

**Purpose:**
- Prevent overfitting
- Save computation time
- Automatic stopping criterion

**Mechanism:**
- Track best validation AUC
- Patience counter: epochs since last improvement
- Threshold: 15 epochs (configurable)

**Patience Preservation:**
- Saved in checkpoint
- Restored on resume
- Auto-restart if exhausted

**Resume Behavior:**
```
If patience_counter >= EARLY_STOPPING_PATIENCE:
    Print warning
    Ignore checkpoint
    Start fresh training (epoch 0, random weights)
Else:
    Load checkpoint
    Continue from saved epoch
    Continue counting patience
```

### Architecture Mismatch Handling

**Problem:**
- Model architecture changes between sessions
- Old checkpoint has incompatible keys
- Standard `load_state_dict` raises error

**Solution:**
- Use `strict=False` parameter
- Load matching keys
- Ignore unexpected keys
- Initialize missing keys randomly
- Print warnings for transparency

**Example:**
```
Checkpoint has: "interaction._weights"
New model has:  "interaction._executor", "interaction._qnode"

Action: Load ligand/pocket encoder weights (compatible)
        Ignore old interaction weights (incompatible)
        Initialize new interaction randomly
```

### Metrics and Evaluation

**Training Metrics:**
- Loss: Binary Cross-Entropy
- Accuracy: Percentage of correct predictions

**Validation Metrics:**
- Loss: BCE on validation set
- Accuracy: Classification accuracy
- AUC: Area Under ROC Curve (primary metric)
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1: Harmonic mean of precision and recall

**Why AUC as Primary Metric:**
- Threshold-independent
- Balanced view of true/false positive trade-off
- Robust to class imbalance
- Clinically relevant for drug screening

---

## Usage Guide

### Basic Training Script

```python
# 1. Setup hardware
from hardware_optimizer import setup_environment
hw_info, config = setup_environment()

# 2. Load data
from data import LigandPocketDataProcessor, LigandPocketDataset, collate_fn
processor = LigandPocketDataProcessor(DATA_DIR, seed=42)
processor.load_data(max_samples=50)
interactions = processor.get_dataset()

# 3. Split data
from sklearn.model_selection import train_test_split
train_ints, temp_ints = train_test_split(interactions, test_size=0.2)
val_ints, test_ints = train_test_split(temp_ints, test_size=0.5)

# 4. Create datasets
train_dataset = LigandPocketDataset(processor, train_ints)
val_dataset = LigandPocketDataset(processor, val_ints)
test_dataset = LigandPocketDataset(processor, test_ints)

# 5. Create dataloaders
from torch.utils.data import DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=config['num_workers'],
    pin_memory=config['pin_memory'],
    persistent_workers=True,
    prefetch_factor=config['prefetch_factor']
)

# 6. Initialize model
from model import LigandPocketQGNN
model = LigandPocketQGNN(
    ligand_in_dim=10,
    pocket_in_dim=19,
    hidden_dim=64,
    n_qubits=6,
    n_qlayers=2,
    use_quantum=True,
    quantum_device=config['quantum_device'],
    use_parallel=True
)

# 7. Train
history, best_auc = train_model(
    model, train_loader, val_loader,
    model_name="quantum",
    device=hw_info['device'],
    resume=True
)
```

### Configuration Parameters

**Data Loading:**
- `max_samples`: Limit number of proteins (for testing)
- `seed`: Random seed for reproducibility

**Model Architecture:**
- `hidden_dim`: GNN/MLP hidden layer size
- `n_qubits`: Quantum circuit width
- `n_qlayers`: Quantum circuit depth
- `use_quantum`: True for quantum, False for classical

**Training:**
- `EPOCHS`: Maximum training epochs (default: 100)
- `LEARNING_RATE`: Adam learning rate (default: 0.001)
- `EARLY_STOPPING_PATIENCE`: Patience threshold (default: 15)

**Hardware:**
- Automatically optimized based on detected hardware
- Can override batch_size, num_workers manually

### Resume Training

**Automatic Resume:**
```python
# If checkpoint exists, automatically resumes
train_model(..., resume=True)
```

**Force Fresh Start:**
```python
# Ignore checkpoint, start from scratch
train_model(..., resume=False)

# OR delete checkpoint files
os.remove("./ligand_pocket_comparison_results/quantum_best.pt")
os.remove("./ligand_pocket_comparison_results/quantum_history.json")
```

**Exhausted Patience Auto-Restart:**
```python
# If checkpoint has patience_counter >= 15
# Automatically starts fresh with warning message
```

### Evaluation

```python
# Load best checkpoint
checkpoint = torch.load("quantum_best.pt")
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Evaluate on test set
test_metrics = evaluate(model, criterion, test_loader, device)
print(f"Test AUC: {test_metrics['auc']:.4f}")
```

---

## Performance Optimization

### Quantum Circuit Optimization

**Lazy Initialization:**
- Defer quantum device creation until first forward pass
- Enables pickling for DataLoader multiprocessing
- Registers nn.Parameter after initialization

**Parallel Evaluation:**
- Standard layer: Sequential sample evaluation
- Parallel layer: ThreadPoolExecutor for batch
- Speedup: ~Nx where N = CPU cores (for batch_size > cores)

**Thread Safety:**
- Move tensors to CPU before threading
- Avoid GPU context conflicts
- Return results to original device

**Small Batch Optimization:**
- batch_size <= 2: Use sequential (avoid threading overhead)
- batch_size > 2: Use parallel

### Data Loading Optimization

**Vectorized Collation:**
- Pre-allocate tensors
- Single-pass concatenation
- Avoid Python loops

**Persistent Workers:**
- Reuse worker processes across epochs
- Avoid repeated spawn overhead
- Trade memory for speed

**Negative Sampling:**
- Batch generation (10,000 candidates at once)
- Filter duplicates vectorized
- Fallback to slower method only if needed

### Memory Management

**Gradient Accumulation (Not Implemented):**
- For very large models or small GPUs
- Simulate larger batch sizes
- Trade computation for memory

**Mixed Precision (Not Implemented):**
- FP16 for forward/backward
- FP32 for optimizer step
- Requires GPU with Tensor Cores

**Gradient Checkpointing (Not Implemented):**
- Recompute activations during backward pass
- Reduce memory footprint
- Trade computation for memory

---

## Troubleshooting

### Common Issues

**1. RuntimeError: Unexpected key(s) in state_dict**

**Cause:** Model architecture changed since checkpoint was saved.

**Solution:**
- Already handled by `strict=False` in `load_checkpoint()`
- Warnings will be printed
- Incompatible layers will be randomly initialized

**2. CUDA Out of Memory**

**Cause:** Batch size too large for GPU memory.

**Solution:**
```python
# Reduce batch size manually
config['batch_size'] = 256  # Or lower

# Or let hardware optimizer handle it
# It caps based on detected memory
```

**3. MPS Sparse Matrix Error**

**Cause:** Apple MPS doesn't fully support sparse matrix multiplication.

**Solution:**
- Already handled in GCNLayer
- Automatically falls back to CPU for sparse ops
- Results moved back to MPS

**4. DataLoader Worker Errors**

**Cause:** Pickling failures with quantum circuits.

**Solution:**
- Lazy initialization in QuantumInteractionLayer
- Defer device creation until forward pass
- Already implemented

**5. Slow Training on CPU**

**Cause:** Small batch size, few workers.

**Solution:**
```python
# Increase batch size (CPU has more RAM)
config['batch_size'] = 2048

# Increase workers
config['num_workers'] = 12

# Enable parallel quantum layer
model = LigandPocketQGNN(..., use_parallel=True)
```

**6. Patience Already Exhausted**

**Cause:** Previous training run reached early stopping.

**Solution:**
- Automatic: Next run will restart fresh
- Manual: Set `resume=False` or delete checkpoint

**7. No Data Found**

**Cause:** Incorrect data directory or file naming.

**Solution:**
```python
# Check directory structure
data/
└── protein_name/
    └── results/
        └── pocket_id/
            ├── *_descriptors_3d.csv  # Must match pattern
            └── *.mol2

# Verify path
processor = LigandPocketDataProcessor("/absolute/path/to/data")
```

**8. Poor Model Performance**

**Possible Causes:**
- Insufficient data (too few samples)
- Poor hyperparameters
- Data quality issues
- Model architecture mismatch

**Debugging Steps:**
1. Check data distribution (balanced pos/neg?)
2. Visualize training curves (overfitting?)
3. Try classical baseline (sanity check)
4. Increase model capacity (hidden_dim, n_qlayers)
5. Adjust learning rate (too high/low?)

### Performance Tuning

**Goal: Maximize GPU Utilization**

1. Profile training loop:
```python
import time
start = time.time()
for epoch in range(10):
    train_epoch(...)
print(f"Time per epoch: {(time.time() - start) / 10:.2f}s")
```

2. Check GPU usage (NVIDIA):
```bash
nvidia-smi -l 1
# Watch GPU utilization percentage
# Goal: >80% utilization
```

3. Adjust batch size:
```python
# Increase until GPU memory ~90% full
# Monitor with nvidia-smi
```

4. Adjust workers:
```python
# Increase if GPU utilization < 80%
# Decrease if CPU usage > 90%
```

**Goal: Minimize Training Time**

1. Use larger hardware:
- Multi-GPU (not implemented, requires DDP)
- Cloud instances (A100, H100)

2. Reduce data size:
- Fewer samples (faster epochs)
- Smaller ligands (faster GNN)

3. Reduce model size:
- Fewer GNN layers
- Smaller hidden_dim
- Fewer quantum layers

4. Early stopping:
- Lower patience threshold
- Accept suboptimal AUC

---

## Appendix

### File Structure

```
QuantumGNN/
├── data.py                   # Data processing pipeline
├── model.py                  # Neural network architectures
├── model_ibm.py             # IBM quantum backend (alternative)
├── hardware_optimizer.py     # Auto hardware detection
├── quantum_parallel.py       # Parallel quantum layer
├── test.ipynb               # Training notebook
├── test_ibm.ipynb           # IBM quantum notebook
├── WIKI.md                  # This documentation
└── ligand_pocket_comparison_results/
    ├── quantum_best.pt      # Quantum model checkpoint
    ├── quantum_history.json # Quantum training history
    ├── classical_best.pt    # Classical model checkpoint
    └── classical_history.json
```

### Glossary

**Ligand:** Small molecule (drug candidate) that may bind to a protein.

**Pocket:** Binding site on protein surface where ligands attach.

**Graph Neural Network (GNN):** Neural network operating on graph-structured data (molecules).

**Quantum Circuit:** Sequence of quantum gates applied to qubits.

**Qubit:** Quantum bit, basic unit of quantum information.

**Entanglement:** Quantum correlation between qubits enabling non-classical computation.

**Expectation Value:** Average measurement outcome of a quantum observable.

**PennyLane:** Python library for quantum machine learning.

**TorchLayer:** PennyLane wrapper making quantum circuits behave like PyTorch modules.

**Sparse Matrix:** Matrix with mostly zero entries (efficient storage/computation).

**COO Format:** Coordinate format for sparse matrices (row indices, column indices, values).

**Batch Normalization:** Not used (GCN uses symmetric normalization instead).

**Early Stopping:** Training termination based on validation performance plateau.

**AUC (Area Under Curve):** Classification metric measuring ROC curve area.

**ROC (Receiver Operating Characteristic):** True positive rate vs false positive rate plot.

### References

**Graph Neural Networks:**
- Kipf & Welling (2017): Semi-Supervised Classification with Graph Convolutional Networks
- Gilmer et al. (2017): Neural Message Passing for Quantum Chemistry

**Quantum Machine Learning:**
- Schuld et al. (2020): Circuit-centric quantum classifiers
- Benedetti et al. (2019): Parameterized quantum circuits as machine learning models

**PennyLane:**
- Bergholm et al. (2018): PennyLane: Automatic differentiation of hybrid quantum-classical computations

**Drug Discovery:**
- Jiménez et al. (2018): KDEEP: Protein-Ligand Absolute Binding Affinity Prediction via 3D-Convolutional Neural Networks
- Torng & Altman (2019): Graph Convolutional Neural Networks for Predicting Drug-Target Interactions

### Contact and Contributions

This project is a research prototype for quantum-classical hybrid machine learning in drug discovery.

**Future Work:**
- Multi-GPU training support
- Hyperparameter optimization (Ray Tune, Optuna)
- Real quantum hardware backend (IBM Quantum, IonQ)
- Larger datasets (PDBBind, ChEMBL)
- Graph attention mechanisms
- 3D protein structure encoding
- Interpretability analysis
- Production deployment pipeline



**Version:** 1.0.0 (Documentation generated 2025-12-14)
