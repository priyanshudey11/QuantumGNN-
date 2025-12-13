# Ligand-Pocket QGNN: Quantum-Enhanced Binding Affinity Prediction

**Ligand-Pocket QGNN** is a hybrid Quantum-Classical Graph Neural Network designed to predict the binding affinity between drug ligands and protein binding pockets. It leverages the expressive power of Quantum Machine Learning (QML) to model complex molecular interactions, comparing performance directly against classical counterparts.

---

## Key Features

*   **Hybrid Architecture:** Combines classical GCNs (for ligands) and MLPs (for protein pockets) with a Variational Quantum Circuit (VQC) interaction layer.
*   **Hardware Acceleration:**
    *   **Apple Silicon (MPS):** Custom parallel processing implementation for PennyLane on Mac GPUs.
    *   **NVIDIA (CUDA):** Full support for `pennylane-lightning-gpu`.
    *   **IBM Quantum:** Direct integration with IBM Quantum hardware via `qiskit-ibm-runtime`.
*   **Parallel Processing:** Custom threaded execution layer (`ParallelQuantumInteractionLayer`) to maximize CPU/GPU throughput for quantum simulations.
*   **Comprehensive Benchmarking:** Dedicated notebooks for comparing Quantum vs. Classical performance metrics (AUC, Accuracy, F1).

---

## System Architecture

The model consists of three main components:

1.  **Ligand Encoder (Classical GNN):**
    *   Input: Molecular graph (atoms & bonds).
    *   Layers: 2x Graph Convolutional Layers (GCN) + Global Pooling.
    *   Output: `n`-dimensional latent vector.

2.  **Pocket Encoder (Classical MLP):**
    *   Input: 3D structural descriptors of the protein pocket.
    *   Output: `n`-dimensional latent vector.

3.  **Interaction Layer (The "Quantum Brain"):**
    *   **Input:** Concatenated Ligand + Pocket vectors.
    *   **Circuit:** Angle Embedding $\to$ Strongly Entangling Layers $\to$ Pauli-Z Measurement.
    *   **Output:** Binding probability (0-1).

---

## How It Works

The pipeline follows a structured data flow to predict binding affinity:

1.  **Data Ingestion:**
    *   **Ligands:** Parsed from MOL2 files into graph structures (Atoms=Nodes, Bonds=Edges).
    *   **Pockets:** Parsed from PDB files into geometric feature vectors (FPocket descriptors).

2.  **Dual-Stream Encoding:**
    *   The **Ligand Graph** is passed through a GCN, aggregating local atomic features into a global molecular representation.
    *   The **Pocket Vector** is processed by a classical MLP to extract high-level geometric embeddings.

3.  **Quantum Fusion:**
    *   The two latent vectors are concatenated and normalized.
    *   This combined vector serves as the input parameters (angles) for the **Variational Quantum Circuit (VQC)**.
    *   The VQC entangles the features in a high-dimensional Hilbert space.

4.  **Prediction:**
    *   A Pauli-Z measurement is performed on the first qubit.
    *   The expectation value is mapped to a probability score representing the binding likelihood.

### Architecture Diagram
```mermaid
graph TD
    subgraph Input
    L[Ligand (MOL2)] --> LG[Ligand Graph]
    P[Pocket (PDB)] --> PV[Pocket Vector]
    end

    subgraph Classical Encoding
    LG --> GCN[GCN Encoder]
    PV --> MLP[MLP Encoder]
    GCN --> LV[Latent Vector L]
    MLP --> PVec[Latent Vector P]
    end

    subgraph Quantum Interaction
    LV & PVec --> C[Concatenation]
    C --> VQC[Variational Quantum Circuit]
    VQC --> M[Measurement]
    end

    M --> Out([Binding Prob])
```

---

## Dataset

This project uses the **CDPPILBP (Comprehensive Dataset of Protein-Protein Interactions and Ligand Binding Pockets)** from Zenodo.

### Dataset Information

**Title:** A Comprehensive Dataset of protein-protein interactions and Ligand Binding Pockets for Advancing Drug Discovery

**Source:** Institut Pasteur, Paris, France
**Published:** November 30, 2023
**Data Collection Date:** March 17, 2023 (from PDBe)
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

**DOI:** [10.5281/zenodo.10805580](https://doi.org/10.5281/zenodo.10805580)
**Paper DOI:** [10.1038/s41597-024-03233-z](https://doi.org/10.1038/s41597-024-03233-z)

### Dataset Statistics

- **~34,475 PDB structures** with protein-protein and protein-ligand complexes
- **~23,000 binding pockets** with 3D geometric descriptors
- **~3,700 unique proteins** across 500+ organisms
- **~3,500 ligands** (small molecules, peptides)
- **Size:** 11.1 GB (compressed)

### Dataset Contents

Each PDB entry includes:
- Full protein structure files (`.ent`, `.pdb`)
- Extracted protein-protein complexes
- Extracted protein-ligand complexes
- Interface residue annotations (6Å cutoff)
- Pre-computed 3D pocket descriptors (~100 features):
  - **Shape descriptors:** Volume, PMI1-3, NPR1-2, Radius of gyration, Asphericity, Eccentricity
  - **Atom type distributions:** CZ, CA, O, OD1, OG, N, NZ counts
  - **Spatial features:** Distance-binned atom counts (40-120Å shells)
- FPocket-generated binding pocket MOL2 files (liganded/unliganded)

### Download Instructions

```bash
# Download dataset (11.1 GB)
wget https://zenodo.org/records/10805580/files/CDPPILBP.tar.gz

# Or use curl
curl -O https://zenodo.org/records/10805580/files/CDPPILBP.tar.gz

# Verify checksum (optional)
md5sum CDPPILBP.tar.gz
# Expected: a2ee28e5dea636e2892712609cd4c215

# Extract
tar -xzf CDPPILBP.tar.gz
```

### Citation

If you use this dataset, please cite:

```bibtex
@article{moine-franel2024comprehensive,
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

@dataset{moine-franel2023zenodo,
  author={Moine-Franel, Alexandra and Mareuil, Fabien and Nilges, Michael and Ciambur, Constantin Bogdan and Sperandio, Olivier},
  title={A Comprehensive Dataset of protein-protein interactions and Ligand Binding Pockets for Advancing Drug Discovery},
  year={2023},
  publisher={Zenodo},
  doi={10.5281/zenodo.10805580},
  url={https://doi.org/10.5281/zenodo.10805580}
}
```

---

## Installation

### Prerequisites
*   Python 3.10+
*   Conda (recommended)

### fast Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/ligand-pocket-qgnn.git
    cd ligand-pocket-qgnn
    ```

2.  **Create Environment:**
    ```bash
    conda create -n quantum python=3.11
    conda activate quantum
    ```

3.  **Install Dependencies:**
    ```bash
    # Install PyTorch (select command for your OS from pytorch.org)
    pip install torch torchvision torchaudio
    
    # Install Project Requirements
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Training & Comparison
Run the main comparison notebook to train both Quantum and Classical models on your local machine:
*   Open `compare_clean.ipynb` or `compare_ligand_pocket_quantum_vs_classical.ipynb`.
*   The notebook will automatically detect your hardware (CPU, CUDA, MPS) and optimize settings.

### 2. Running on IBM Quantum
To run on real quantum hardware or IBM simulators:
*   Open `compare_ligand_pocket_quantum_vs_classical_IBM.ipynb`.
*   Configure your IBM Quantum Token within the notebook.

### 3. Hardware Optimization
The system includes `hardware_optimizer.py` which automatically configures:
*   `num_workers` for DataLoader.
*   `batch_size` based on VRAM/RAM.
*   PennyLane device backend (`lightning.qubit`, `default.qubit`, etc.).

---

## Project Structure

```plaintext
├── ligand_pocket_qgnn/          # Main Package
│   ├── model.py                 # Core QGNN Architecture (Local Simulation)
│   ├── model_ibm.py             # QGNN Architecture (IBM Runtime Integration)
│   ├── quantum_parallel.py      # Thread-Parallel Quantum Layer
│   ├── data.py                  # Data Loading & Graph Processing
│   └── hardware_optimizer.py    # Auto-configuration Utility
│   └── test.ipynb               # Main Training & Benchmarking Notebook
│   └── requirements.txt         # Project Dependencies
└── README.md                    # This file
```

---

## Technical Highlights

### Solving the "MPS Bottleneck"
Running quantum simulations on Apple Silicon (MPS) is traditionally challenging due to lack of native sparse tensor support and thread safety issues in simulators.
*   **Solution:** We implemented `ParallelQuantumInteractionLayer` which intelligently offloads quantum circuit evaluation to CPU threads while keeping the rest of the pipeline on the GPU, preventing `AssertionError` crashes and maximizing throughput.

### IBM Runtime Integration
Uses `qiskit-ibm-runtime` primitives (`Estimator`, `Sampler`) for efficient session-based execution on IBM hardware, reducing queue times and latency.

---

## Citation

If you use this code in your research, please cite:

**This Work:**
> Priyanshu, "Ligand-Pocket QGNN: Exploring Quantum Advantage in Drug Discovery," 2025.

**Dataset:**
> Moine-Franel, A., Mareuil, F., Nilges, M., Ciambur, C.B., & Sperandio, O. (2024). A comprehensive dataset of protein-protein interactions and ligand binding pockets for advancing drug discovery. *Scientific Data*, 11, 402. https://doi.org/10.1038/s41597-024-03233-z

---

## Acknowledgments

- **Dataset:** Institut Pasteur (Structural Bioinformatics Unit) for the CDPPILBP dataset
- **Quantum Computing:** IBM Quantum for hardware access
- **Frameworks:** PennyLane, PyTorch, PyTorch Geometric
