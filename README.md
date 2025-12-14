# Ligand-Pocket QGNN: Hybrid Quantum-Classical Graph Neural Network for Drug-Protein Interaction Prediction

A hybrid quantum-classical graph neural network for predicting drug-protein binding interactions. Compare quantum circuits vs. classical neural networks on molecular data with automatic hardware optimization.

---

## Quick Start

**New to this project?** Start here: [GETTING_STARTED.md](GETTING_STARTED.md)

```bash
# 1. Install
conda create -n quantum python=3.11 -y && conda activate quantum
pip install torch pennylane scikit-learn pandas numpy tqdm matplotlib

# 2. Run
jupyter notebook test.ipynb  # Run all cells
```

This trains both quantum and classical models and generates comparison plots.

**Full documentation**: [WIKI.md](WIKI.md)

---

## What This Does

Compares quantum vs. classical approaches for predicting drug-protein binding:
- **Ligand Encoder**: Graph Neural Network (GCN) processes molecular structure
- **Pocket Encoder**: MLP processes protein binding site features
- **Interaction Layer**: 6-qubit quantum circuit OR classical neural network
- **Output**: Binding probability prediction

Uses **CDPPILBP** dataset (34k+ protein-ligand structures). Download: [Zenodo](https://zenodo.org/records/10805580)

---

## Documentation

- **[WIKI.md](WIKI.md)**: Complete API documentation and technical details

---

## Project Structure

```
QuantumGNN/
├── data.py                  # Data loading and graph construction
├── model.py                 # QGNN and classical architectures
├── hardware_optimizer.py    # Auto hardware detection
├── test.ipynb              # Main training notebook
├── requirements.txt        # Dependencies
├── GETTING_STARTED.md      # Setup guide
└── WIKI.md                 # Full documentation
```

---

## Citation

```
Dey, Priyanshu. "A Hybrid Quantum Graph Neural Network Methodology for
Drug-Protein Interaction Simulation." Pennsylvania State University, 2025.
```

Dataset: Moine-Franel et al. (2024). [doi:10.1038/s41597-024-03233-z](https://doi.org/10.1038/s41597-024-03233-z)

---

## License

MIT License. CDPPILBP dataset: CC BY 4.0

**Author**: Priyanshu Dey | **Email**: pkd5228@psu.edu | **Institution**: Pennsylvania State University

