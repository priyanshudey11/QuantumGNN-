
## Quick Setup 

### Step 1: Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/QuantumGNN.git
cd QuantumGNN

# Create a fresh Python environment
conda create -n quantum python=3.11 -y
conda activate quantum

# Install PyTorch (choose one based on your system)
# For CPU only:
pip install torch torchvision torchaudio

# For NVIDIA GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For macOS (Apple Silicon):
pip install torch torchvision torchaudio

# Install remaining dependencies
pip install -r requirements.txt
```

### Step 2: Get Sample Data

For a quick test, the code includes a small sample mode that uses only 50 protein structures:

```python
# This is already configured in test.ipynb
processor.load_data(max_samples=50)  # Fast: ~10 seconds, 728 interactions
```

To use the full CDPPILBP dataset (optional, for production use):

```bash
# Download
wget https://zenodo.org/records/10805580/files/CDPPILBP.tar.gz

# Extract (90 GB uncompressed)
tar -xzf CDPPILBP.tar.gz

# Update data path in test.ipynb
DATA_DIR = "/path/to/extracted/CDPPILBP"
```

# Ligand-Pocket QGNN: Hybrid Quantum-Classical Graph Neural Network for Drug-Protein Interaction Prediction

A research implementation of a hybrid quantum-classical graph neural network for predicting ligand-pocket binding interactions in drug discovery. This project provides a complete methodology for constructing, training, and evaluating QGNN architectures on molecular data with comprehensive hardware optimization.

---

## Overview

### Purpose

This framework implements a hybrid quantum-classical architecture that combines:
- Classical Graph Neural Networks for ligand molecular structure encoding
- Classical Multi-Layer Perceptrons for protein pocket feature extraction
- Variational Quantum Circuits for interaction modeling

The system enables direct comparison between quantum and classical interaction mechanisms while maintaining identical preprocessing, encoding, and evaluation pipelines.

### Key Capabilities

- **Hardware-Agnostic Execution**: Automatic detection and optimization for CUDA (NVIDIA), MPS (Apple Silicon), and CPU platforms
- **Quantum Circuit Simulation**: 6-qubit, 2-layer variational quantum circuits with angle encoding and strongly entangling layers
- **Parallel Processing**: Multi-threaded quantum circuit evaluation for improved training throughput on multi-core systems
- **Checkpoint Management**: Patience-aware training resumption with automatic architecture mismatch handling
- **Comprehensive Evaluation**: Full suite of classification metrics (AUC, accuracy, precision, recall, F1-score)

---

## Architecture

### System Components

```
Input Layer
├── Ligand Processing: MOL2 → Graph (atoms as nodes, bonds as edges)
└── Pocket Processing: CSV descriptors → Feature vector (19 dimensions)

Encoding Layer
├── Ligand Encoder: 2-layer GCN + Global Mean Pooling → 3D latent vector
└── Pocket Encoder: 3-layer MLP → 3D latent vector

Interaction Layer (Switchable)
├── Quantum: 6-qubit VQC (angle encoding + strongly entangling + measurement)
└── Classical: 2-layer MLP with ReLU and Sigmoid

Output Layer
└── Binary Classification: Binding probability (0-1)
```

### Data Flow

```
Ligand Graph (N atoms, E edges)     Pocket Features (19-dim vector)
          ↓                                      ↓
    LigandGNN (GCN)                        PocketMLP
          ↓                                      ↓
    3D embedding                           3D embedding
          ↓                                      ↓
          └──────────── Concatenation ──────────┘
                          ↓
                    6D combined vector
                          ↓
            Normalization: tanh(x) * π
                          ↓
          ┌───────────────┴───────────────┐
          ↓                               ↓
    Quantum Circuit                  Classical MLP
    (6Q/2L VQC)                     (64-dim hidden)
          ↓                               ↓
    Pauli-Z measurement             Sigmoid activation
          ↓                               ↓
          └──────── Binding Score ────────┘
                    (probability)
```

---

## Dataset

### CDPPILBP: Comprehensive Dataset of Protein-Protein Interactions and Ligand Binding Pockets

**Source**: Institut Pasteur, Paris (Structural Bioinformatics Unit)
**License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
**Publication**: Scientific Data, Nature Publishing Group (April 2024)

#### Dataset Statistics

- 34,475 PDB structures with experimentally determined complexes
- 23,000 annotated binding pockets with 3D geometric descriptors
- 3,700 unique proteins spanning 500+ organisms
- 3,500 distinct ligands (small molecules, drug-like compounds, peptides, cofactors)
- 11.1 GB compressed (90 GB uncompressed)

#### Data Contents

Each protein entry includes:

1. **Structural Files**
   - PDB format coordinates (.pdb, .ent)
   - Extracted protein-ligand complexes
   - Interface annotations (6 Angstrom cutoff)

2. **Pocket Descriptors** (100+ pre-computed features via FPocket)
   - **Geometric**: Volume, PMI1-3, NPR1-2, radius of gyration, asphericity, spherocity index, eccentricity, inertial shape factor
   - **Chemical**: Atom type distributions (CZ, CA, O, OD1, OG, N, NZ, DU)
   - **Spatial**: Distance-binned atom counts (40-120 Angstrom shells)

3. **Ligand Structures**
   - MOL2 format binding pocket files
   - Both liganded and unliganded states

#### Access and Citation

**Download**: [Zenodo Repository](https://zenodo.org/records/10805580)
**DOI**: 10.5281/zenodo.10805580
**MD5 Checksum**: a2ee28e5dea636e2892712609cd4c215

```bash
# Download and verify dataset
wget https://zenodo.org/records/10805580/files/CDPPILBP.tar.gz
md5sum CDPPILBP.tar.gz  # Should match: a2ee28e5dea636e2892712609cd4c215
tar -xzf CDPPILBP.tar.gz
```

**Citation**:
```bibtex
@article{moine2024comprehensive,
  title={A comprehensive dataset of protein-protein interactions and ligand binding pockets for advancing drug discovery},
  author={Moine-Franel, Alexandra and Mareuil, Fabien and Nilges, Michael and Ciambur, Constantin Bogdan and Sperandio, Olivier},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={402},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s41597-024-03233-z}
}
```

---

## Installation

### Requirements

- Python 3.10 or higher
- CUDA 11.7+ (optional, for NVIDIA GPU support)
- 8 GB RAM minimum (16 GB+ recommended)

### Setup Instructions

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/QuantumGNN.git
   cd QuantumGNN
   ```

2. **Create Environment**
   ```bash
   conda create -n quantum python=3.11
   conda activate quantum
   ```

3. **Install PyTorch**

   Select appropriate command from [pytorch.org](https://pytorch.org/get-started/locally/):

   ```bash
   # CPU-only
   pip install torch torchvision torchaudio

   # CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # macOS (MPS)
   pip install torch torchvision torchaudio
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Core Dependencies

- **PyTorch** (2.0+): Deep learning framework
- **PennyLane** (0.32+): Quantum machine learning library
- **PyTorch Geometric** (2.3+): Graph neural network extensions
- **NumPy, Pandas**: Data manipulation
- **scikit-learn**: Train/test splitting and metrics
- **tqdm**: Progress bars

---

## Quick Start

### Basic Training Example

```python
from hardware_optimizer import setup_environment
from data import LigandPocketDataProcessor, LigandPocketDataset, collate_fn
from model import LigandPocketQGNN
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# 1. Auto-detect and optimize for hardware
hw_info, config = setup_environment()

# 2. Load data
processor = LigandPocketDataProcessor(data_dir="/path/to/CDPPILBP", seed=42)
processor.load_data(max_samples=None)  # Load all data
interactions = processor.get_dataset()

# 3. Split data (80/10/10)
train_ints, temp_ints = train_test_split(interactions, test_size=0.2, random_state=42)
val_ints, test_ints = train_test_split(temp_ints, test_size=0.5, random_state=42)

# 4. Create datasets and dataloaders
train_dataset = LigandPocketDataset(processor, train_ints)
val_dataset = LigandPocketDataset(processor, val_ints)

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

# 5. Initialize model
quantum_model = LigandPocketQGNN(
    ligand_in_dim=10,          # Atom type one-hot encoding
    pocket_in_dim=19,          # Pocket geometric + atom count features
    hidden_dim=64,             # GNN/MLP hidden dimension
    n_qubits=6,                # Quantum circuit width
    n_qlayers=2,               # Quantum circuit depth
    use_quantum=True,          # True for QGNN, False for classical
    quantum_device=config['quantum_device'],
    use_parallel=True          # Enable multi-threaded quantum evaluation
)

# 6. Train (see test.ipynb for full training loop)
from train import train_model
history, best_auc = train_model(
    quantum_model, train_loader, val_loader,
    model_name="quantum",
    device=hw_info['device'],
    resume=True  # Automatically resume from checkpoint if available
)
```

### Running Comparison Experiments

The project includes Jupyter notebooks for comparative evaluation:

1. **Local Simulation**: `test.ipynb`
   - Trains both quantum and classical models
   - Automatic hardware optimization
   - Generates performance comparison plots

2. **IBM Quantum Hardware**: `test_ibm.ipynb`
   - Integrates with IBM Quantum via qiskit-ibm-runtime
   - Supports Qiskit Runtime Estimator and Sampler primitives
   - Requires IBM Quantum account and API token

---

## Model Configuration

### Quantum Circuit Design

**Architecture**: 6-qubit, 2-layer Variational Quantum Circuit (VQC)

**Components**:
1. **Encoding**: Angle embedding via RY rotations
   - Maps 6D classical vector to quantum state: |ψ⟩ = ⊗ RY(θᵢ)|0⟩

2. **Ansatz**: Strongly entangling layers
   - Per layer: Single-qubit rotations (RZ-RY-RZ) + CNOT entanglers
   - Entanglement pattern: Full connectivity via successive CX gates

3. **Measurement**: Pauli-Z expectation value on qubit 0
   - Output ∈ [-1, 1], mapped to [0, 1] via linear transformation

**Trainable Parameters**: 36 (2 layers × 6 qubits × 3 rotation angles)

**Gradient Computation**: Parameter-shift rule (exact gradients for quantum gates)

### Classical Baseline

**Architecture**: 2-layer MLP

```
Input (6D) → Linear(64) → ReLU → Linear(1) → Sigmoid → Output
```

**Trainable Parameters**: 513 (6×64 + 64 + 64×1 + 1)

### Hardware Optimization

The `hardware_optimizer.py` module automatically configures:

| Hardware | Batch Size | Workers | Prefetch | Quantum Backend |
|----------|------------|---------|----------|-----------------|
| NVIDIA A100/H100 | 4096 | 12 | 6 | lightning.gpu |
| NVIDIA RTX 4090 | 2048 | 8 | 4 | lightning.gpu |
| Apple M4 | 1536 | 10 | 5 | lightning.qubit |
| Apple M3 | 1024 | 8 | 4 | lightning.qubit |
| AMD EPYC | 3072 | 16 | 8 | lightning.qubit |
| Intel Xeon | 2048 | 12 | 6 | lightning.qubit |

Batch size and worker count are further adjusted based on available memory.

---

## Training Features

### Checkpoint Management

- **Automatic Saving**: Best model saved based on validation AUC
- **Patience Tracking**: Early stopping after 15 epochs without improvement
- **Resume Support**: Automatically resume from last checkpoint with `resume=True`
- **Architecture Mismatch Handling**: Loads compatible weights with `strict=False`, initializes incompatible layers randomly
- **Auto-Restart**: If patience exhausted, automatically starts fresh training

### Optimization

- **Optimizer**: Adam (learning rate: 1e-3)
- **Loss Function**: Binary Cross-Entropy
- **Gradient Clipping**: max_norm=1.0 for stability
- **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early Stopping**: Monitors validation AUC with patience=15

### Metrics

**Training**: Loss, Accuracy
**Validation**: Loss, Accuracy, AUC, Precision, Recall, F1-score
**Primary Metric**: AUC (threshold-independent, robust to class imbalance)

---

## Performance Considerations

### Computational Requirements

**Quantum Model**:
- Training time: ~2-5 seconds per epoch (batch_size=1024, 728 samples, Apple M4)
- Memory: ~2-4 GB (model + data + quantum simulation overhead)
- Bottleneck: Quantum circuit evaluation (scales exponentially with qubits in classical simulation)

**Classical Model**:
- Training time: ~0.5-1 second per epoch (same configuration)
- Memory: ~1-2 GB
- Significantly faster per epoch, but quantum may achieve comparable performance with fewer parameters

### Parallelization

**ParallelQuantumInteractionLayer**:
- Evaluates quantum circuits concurrently across CPU threads
- Speedup: ~Nx where N = CPU core count (for batch_size > cores)
- Automatically disabled for batch_size ≤ 2 to avoid threading overhead

**DataLoader Multi-Processing**:
- Persistent workers avoid repeated process spawning
- Prefetching overlaps data loading with GPU computation
- Optimal worker count: 4-12 depending on CPU cores

---

## Project Structure

```
QuantumGNN/
├── data.py                          # Data loading and graph construction
├── model.py                         # QGNN and classical GNN architectures
├── model_ibm.py                     # IBM Quantum hardware integration
├── quantum_parallel.py              # Multi-threaded quantum layer
├── hardware_optimizer.py            # Auto hardware detection and config
├── test.ipynb                       # Main training and comparison notebook
├── test_ibm.ipynb                   # IBM Quantum hardware notebook
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── WIKI.md                          # Comprehensive technical documentation
├── PAPER_CODE_ALIGNMENT.md          # Analysis of paper vs. code correspondence
└── ligand_pocket_comparison_results/
    ├── quantum_best.pt              # Best quantum model checkpoint
    ├── quantum_history.json         # Training history (quantum)
    ├── classical_best.pt            # Best classical model checkpoint
    └── classical_history.json       # Training history (classical)
```

---

## Documentation

- **[WIKI.md](WIKI.md)**: Comprehensive technical documentation covering:
  - Module-level API documentation
  - Data pipeline details
  - Model architecture deep-dive
  - Hardware optimization strategies
  - Training workflow and checkpoint management
  - Troubleshooting guide

- **[PAPER_CODE_ALIGNMENT.md](PAPER_CODE_ALIGNMENT.md)**: Detailed analysis of correspondence between research paper claims and code implementation

---

## Technical Highlights

### MPS (Apple Silicon) Optimization

**Challenge**: Apple MPS backend lacks native support for sparse matrix operations and has thread safety constraints in quantum simulators.

**Solution**:
- GCNLayer automatically falls back to CPU for sparse operations, then moves results back to MPS
- ParallelQuantumInteractionLayer moves tensors to CPU for thread-safe quantum evaluation, returns to original device
- Prevents AssertionError crashes while maintaining GPU utilization for other operations

### Quantum Circuit Lazy Initialization

**Problem**: PennyLane quantum devices cannot be pickled for DataLoader multiprocessing.

**Solution**:
- Defer quantum device creation until first forward pass
- Enables DataLoader persistent workers without serialization errors
- Registers nn.Parameter weights after device initialization

### Graph Batching

**Challenge**: Variable-size molecular graphs cannot be stacked into tensors directly.

**Solution**:
- Concatenate all graphs into a single large graph
- Shift edge indices by cumulative node offsets to maintain connectivity
- Track batch assignment vector for graph-level pooling
- O(1) memory overhead, efficient GPU execution

---

## Limitations and Known Issues

1. **Scale**: Current implementation tested on subsets (50-1000 protein structures). Full dataset experiments require 32+ GB RAM.

2. **Quantum Simulation**: 6-qubit circuits simulated classically. Scales exponentially (O(2^n) complexity). Real quantum hardware integration via IBM Quantum available but subject to queue times and device noise.

3. **Graph Structure**: Ligand-only graphs. Paper describes bipartite ligand-pocket graphs, not yet implemented.

4. **Feature Set**: Uses 10-dim ligand features (atom types) and 19-dim pocket features (11 geometric + 8 atom counts). Paper describes 100+ pocket descriptors.

5. **Architecture**: Implements GCN (Kipf & Welling), not MPNN-style message passing described in associated paper.

6. **Performance Gap**: Classical model often outperforms quantum in final validation metrics, suggesting shallow quantum circuits (6Q/2L) may be insufficient for this task complexity.

---

## Future Work

- [ ] Bipartite graph construction (ligand-pocket edges)
- [ ] Extended feature extraction (100+ pocket descriptors)
- [ ] MPNN-style message passing implementation
- [ ] Multi-qubit measurement (currently measures only qubit 0)
- [ ] Deeper quantum circuits (4-6 layers) and/or more qubits (8-12)
- [ ] Hybrid training (classical pre-training + quantum fine-tuning)
- [ ] Real quantum hardware benchmarking (IBM, IonQ, Rigetti)
- [ ] Multi-GPU training support (DistributedDataParallel)
- [ ] Hyperparameter optimization (Ray Tune, Optuna)
- [ ] Graph attention mechanisms
- [ ] 3D protein structure encoding (beyond pocket descriptors)

---


**Dataset**:
```bibtex
@article{moine2024comprehensive,
  title={A comprehensive dataset of protein-protein interactions and ligand binding
         pockets for advancing drug discovery},
  author={Moine-Franel, Alexandra and Mareuil, Fabien and Nilges, Michael and
          Ciambur, Constantin Bogdan and Sperandio, Olivier},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={402},
  year={2024},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41597-024-03233-z}
}
```




**Version**: 1.0.0
**Last Updated**: December 14, 2025
