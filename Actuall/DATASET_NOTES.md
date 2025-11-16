# Dataset Notes: Protein-Protein Interaction Dataset

## Overview

Your dataset comes from the Nature Scientific Data paper:
**"A comprehensive dataset of protein-protein interactions and ligand binding pockets for advancing drug discovery"** (2024)

## Dataset Statistics

- **23,000+ pockets** from protein-protein interactions
- **3,700+ proteins** across 500+ organisms
- **1,700+ unique protein families** (Pfam domains)
- **~3,500 ligands** with binding information

## Dataset Subsets

### 1. HD (Heterodimer) Dataset
- **4,770 PDB structures**
- **18,266 pockets**
- Protein-protein interaction binding sites
- Used as "negative" examples (no small molecule ligands)

### 2. PLOC (Protein-Ligand Orthosteric Competitive)
- **1,863 PDB structures**
- **2,325 pockets**
- **1,647 unique ligands**
- Ligands that **compete** with protein partner at interface
- Best for training drug discovery models

### 3. PLONC (Protein-Ligand Orthosteric Non-Competitive)
- **715 PDB structures**
- **817 pockets**
- **539 unique ligands**
- Ligands at interface but **don't compete** with protein partner

### 4. PLA (Protein-Ligand Allosteric)
- **1,613 PDB structures**
- **1,830 pockets**
- **1,277 unique ligands**
- Ligands **away from** the interface (allosteric sites)

## Data Structure

Each PDB entry (e.g., `1a0n`) contains:

```
1a0n/
├── pdb1a0n.ent                                    # Raw PDB file
├── 1a0n--AB--P27986--P06241.pdb                  # Protein-protein complex
├── 1a0n--AB--P27986__interface-residues_6A.txt   # Interface residues
└── results/
    └── 1a0n--A--P27986__Repair-H_descriptors_3d.csv  # 3D DESCRIPTORS ⭐
```

## 3D Descriptor Features (CSV Files)

The `*_descriptors_3d.csv` files contain **109 features** per pocket:

### Geometric Descriptors (from RDKit3D):
- `Volume`: Binding pocket volume
- `PMI1, PMI2, PMI3`: Principal moments of inertia (shape)
- `NPR1, NPR2`: Normalized principal moment ratios
- `Rgyr`: Radius of gyration
- `Asphericity`: Deviation from sphere
- `SpherocityIndex`: Sphericity measure
- `Eccentricity`: Elongation measure
- `InertialShapeFactor`: Mass distribution

### Atom Type Counts:
- `CZ`: Aromatic carbon
- `CA`: Aliphatic carbon
- `O`: Oxygen
- `OD1`: Carboxyl oxygen
- `OG`: Hydroxyl oxygen
- `N`: Nitrogen
- `NZ`: Charged nitrogen
- `DU`: Dummy atoms

### Burial/Exposure Features:
- `CZ40`, `CZ50`, ..., `CZ120`: Carbon counts at different burial depths
- `T40`, `T50`, ..., `T120`: Total atoms at different depths
- Similar for all atom types

## Key Features for Drug-Patient QGNN

For your **Drug-Patient QGNN pipeline**, you should use:

### Drug Features (Molecular Descriptors):
- Load from: `results/*_descriptors_3d.csv`
- Use columns: `Volume, PMI1, PMI2, PMI3, NPR1, NPR2, Rgyr, Asphericity, SpherocityIndex, Eccentricity, InertialShapeFactor`
- Plus atom counts: `CZ, CA, O, OD1, OG, N, NZ, DU`
- **Total: 19 features** (matches your current implementation with 11 geometric + 8 atom features)

### Recommended Usage:

1. **For Drug Nodes**:
   - Load PLOC dataset (competitive inhibitors)
   - Extract 3D descriptors from CSV files
   - These represent "druggable" binding pockets

2. **For Patient Nodes**:
   - Continue using synthetic patient data OR
   - Load real EHR/clinical data

3. **For Interactions**:
   - Ligand binding affinity → efficacy proxy
   - Pocket properties → drug response modulation

## File Naming Convention

Example: `3hwe--A--P80188__Repair-H_descriptors_3d.csv`

- `3hwe`: PDB code
- `A`: Chain identifier
- `P80188`: UniProt ID
- `Repair-H`: Hydrogens added by FoldX
- `descriptors_3d`: 3D geometric descriptors

## How Your Pipeline Uses This

Your `DrugPatientDataProcessor.load_protein_ligand_data()` function:

1. Searches for: `{data_dir}/**/results/*_descriptors_3d.csv`
2. Extracts features: Volume, PMI1-3, NPR1-2, Rgyr, shape descriptors, atom counts
3. Creates drug nodes with these molecular features
4. Pairs with synthetic patient nodes
5. Trains QGNN to predict drug-patient interaction outcomes

## Next Steps

1. ✅ Your implementation correctly loads these CSV files
2. ✅ Feature extraction matches the dataset structure
3. 🎯 Consider adding ligand binding affinity data if available
4. 🎯 Could extend to use PFAM/CATH annotations for drug family grouping

## References

- Dataset DOI: https://doi.org/10.5281/zenodo.10805580
- Paper: Moine-Franel et al., Scientific Data (2024)
- Data location: `/media/priyanshu/SD/othercode/data/`
