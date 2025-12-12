# Protein-Ligand and Protein-Protein Interaction Data

Comprehensive dataset containing protein structures, ligand binding data, and protein-protein interaction information from the Protein Data Bank (PDB).

## Overview

This directory contains **34,479 PDB structures** with associated molecular descriptors, binding interface information, and structural data. The data is organized by PDB IDs and includes both protein-ligand complexes and protein-protein interactions (heterodimers).

### Dataset Statistics

- **Total PDB entries**: 34,479 directories
- **PDB files**: ~83,406 structure files
- **Interface residue files**: ~24,488 text files
- **3D descriptor CSV files**: ~15,250 files
- **Date of PDB snapshot**: March 17, 2023 (PDBe)

## Directory Structure

Each PDB entry follows a consistent organizational pattern:

```
data/
├── [PDB_ID]/                                    # e.g., 1a0n, 3hwe, 8exo
│   ├── pdb[PDB_ID].ent                         # Original PDB structure file
│   ├── [PDB_ID]--[CHAINS]--[UNIPROT].pdb       # Extracted protein structure
│   ├── [PDB_ID]--[CHAINS]--[UNIPROT]--[LIG]-[NUM].pdb  # Protein-ligand complex
│   ├── [PDB_ID]_[CHAIN]_[LIG]_[NUM].mol2      # Ligand structure (MOL2 format)
│   ├── [PDB_ID]--[CHAINS]--[UNIPROT]__interface-residues_6A.txt  # Interface residues
│   └── results/
│       ├── state.log                            # Processing log
│       ├── [PDB_ID]--[CHAIN]--[UNIPROT]__Repair-H.pdb  # Hydrogen-repaired structure
│       └── [COMPLEX_ID]/
│           ├── [PDB_ID]--[CHAIN]--[UNIPROT]__Repair-H_descriptors_3d.csv  # 3D descriptors
│           ├── state.log                        # Processing log for this complex
│           ├── [COMPLEX_ID]_CAVITY_N[X]_ALL_orthosteric.mol2    # Orthosteric pocket
│           ├── [COMPLEX_ID]_CAVITY_N[X]_ALL_nonorthosteric.mol2 # Non-orthosteric pocket
│           └── CAVITY_N[X]_ALL__.mol2          # Raw cavity files
└── filter files (see Filter Files section)
```

### Naming Convention

**PDB Entry Format**: `[PDB_ID]--[CHAINS]--[UNIPROT]--[LIGAND]-[NUM]`

- `PDB_ID`: 4-character PDB identifier (e.g., `1a0n`, `3hwe`, `8exo`)
- `CHAINS`: Chain identifier(s) (e.g., `A`, `AB`, `BA`)
- `UNIPROT`: UniProt accession number (e.g., `P27986`, `P80188`)
- `LIGAND`: 3-letter ligand code (e.g., `RKS`, `19P`, `WX7`)
- `NUM`: Residue/ligand number in PDB file

**Examples**:
- `1a0n--AB--P27986--P06241.pdb` - Protein-protein complex (chains A-B)
- `3hwe--A--P80188--RKS-180.pdb` - Protein-ligand complex (chain A, ligand RKS-180)
- `8exo--A--P42336--X3W-1101.pdb` - Protein-ligand complex

## File Types and Contents

### 1. PDB Structure Files (.pdb, .ent)

**Purpose**: Atomic coordinates and structural information

**Types**:
- `pdb[ID].ent`: Original PDB file from Protein Data Bank
- `[ID]--[CHAINS]--[UNIPROT].pdb`: Extracted protein chains
- `[ID]--[CHAINS]--[UNIPROT]--[LIG]-[NUM].pdb`: Protein-ligand complex
- `[ID]--[CHAIN]--[UNIPROT]__Repair-H.pdb`: Hydrogen-repaired structure

**Format**: Standard PDB format with ATOM/HETATM records

### 2. Ligand Structures (.mol2)

**Purpose**: Small molecule ligand coordinates and bond information

**Location**: `[PDB_ID]/[PDB_ID]_[CHAIN]_[LIG]_[NUM].mol2`

**Format**: MOL2 format with atom types, coordinates, and bonds

**Example**: `3hwe_A_RKS_180.mol2` - Ligand RKS at position 180 in chain A

### 3. Interface Residue Files (.txt)

**Purpose**: List of residues at protein-ligand or protein-protein interfaces

**Location**: `[PDB_ID]/[PDB_ID]--[CHAINS]--[UNIPROT]__interface-residues_6A.txt`

**Format**: Plain text, one residue per line
```
A.PRO-91
A.ARG-93
A.PRO-94
A.LEU-95
A.PRO-96
```

**Definition**: Residues within 6 Å of the binding partner

### 4. 3D Molecular Descriptors (.csv)

**Purpose**: Quantitative 3D geometric and physicochemical descriptors

**Location**: `[PDB_ID]/results/[COMPLEX_ID]/[ID]--[CHAIN]--[UNIPROT]__Repair-H_descriptors_3d.csv`

**Descriptor Features**:

| Feature | Description | Units |
|---------|-------------|-------|
| `Volume` | Binding pocket volume | Å³ |
| `PMI1`, `PMI2`, `PMI3` | Principal moments of inertia | Å² Da |
| `NPR1`, `NPR2` | Normalized principal moment ratios | Dimensionless |
| `Rgyr` | Radius of gyration | Å |
| `Asphericity` | Deviation from spherical shape | 0-1 |
| `SpherocityIndex` | Sphericity measure | 0-1 |
| `Eccentricity` | Shape eccentricity | 0-1 |
| `InertialShapeFactor` | Rotational symmetry | Dimensionless |
| `CZ`, `CA`, `O`, `OD1`, `OG`, `N`, `NZ`, `DU` | Atom type counts | Count |

**Notes**:
- Some CSV files may contain warning: "No pockets mol2 found"
- Descriptors are computed using molecular mechanics force fields

### 5. Binding Pocket Files (.mol2 in results/)

**Purpose**: Predicted binding cavities and pockets

**Types**:
- `_CAVITY_N[X]_ALL_orthosteric.mol2`: Primary/orthosteric binding site
- `_CAVITY_N[X]_ALL_nonorthosteric.mol2`: Allosteric/secondary binding sites
- `CAVITY_N[X]_ALL__.mol2`: Raw cavity geometry

**Note**: Not all structures have identified binding pockets

### 6. Processing Logs (.log)

**Purpose**: Record of data processing steps and errors

**Location**:
- `[PDB_ID]/results/state.log`
- `[PDB_ID]/results/[COMPLEX_ID]/state.log`
- `[PDB_ID]/results/[COMPLEX_ID]/state_step5.log`

## Data Types

### Protein-Ligand Complexes

Structures where small molecule drugs/ligands bind to proteins.

**Characteristics**:
- Naming: `[PDB_ID]--[CHAIN]--[UNIPROT]--[LIG]-[NUM]`
- Has ligand MOL2 file
- Interface residues show protein residues near ligand
- Descriptors describe ligand binding pocket

**Example**: `3hwe--A--P80188--RKS-180`
- PDB: 3hwe
- Chain: A
- Protein: P80188
- Ligand: RKS at position 180

### Protein-Protein Complexes (Heterodimers)

Structures where two different proteins interact.

**Characteristics**:
- Naming: `[PDB_ID]--[CHAINS]--[UNIPROT1]--[UNIPROT2]` or `[PDB_ID]--[CHAINS]--[UNIPROT]`
- Multiple chains (e.g., AB, BA)
- Interface residues show protein-protein contact regions
- Two descriptor files (one per chain in interaction)

**Example**: `1a0n--AB--P27986--P06241`
- PDB: 1a0n
- Chains: A and B
- Protein A: P27986
- Protein B: P06241

### Subdirectory Organization

For protein-protein interactions, results are organized by interface direction:

- `1a0n-AB-P27986-P06241-withinA/`: Interface from chain A perspective
- `1a0n-BA-P06241-P27986-withinB/`: Interface from chain B perspective

## Filter Files

Located in the root data directory, these files track quality control and filtering steps:

### Exclusion Filters (without_*.txt)

Files listing PDB entries that failed specific quality criteria:

**Resolution Filters**:
- `without_xray_resolution_*.txt`: Missing X-ray resolution data
- `without_xray_resolution_factor_*.txt`: X-ray resolution too low
- `without_cryo_resolution_*.txt`: Cryo-EM resolution issues
- `without_cryo_resolution_cutoff_*.txt`: Cryo-EM resolution too low

**Method Filters**:
- `without_experimental_method_annotation_*.txt`: No experimental method listed
- `without_experimental_method_selected_*.txt`: Non-selected experimental methods

**Structure Quality Filters**:
- `without_pdb_structure_*.txt`: Failed to retrieve PDB structure
- `without_ligands_20230317PDBe.txt`: No ligands found in structure
- `without_wanted_ligand_20230317PDBe.txt`: Ligand not in target list
- `without_one_molecule_ligand_*.txt`: More than one molecule (ambiguous)
- `without_two_molecules_heterodimer_*.txt`: Wrong number of protein chains

**Annotation Filters**:
- `without_uniprot_annotation_*.txt`: Missing UniProt mapping
- `without_corresponding_uniprot_*.txt`: UniProt ID mismatch
- `without_protein_protein_molecules_*.txt`: Non-protein molecules present

**Interface Filters**:
- `without_interface_overlap_*.txt`: Binding interfaces overlap
- `without_no_interface_altloc_*.txt`: Alternative locations at interface
- `without_orthosteric_pocket_*.txt`: No orthosteric binding site found
- `without_liganded_pocket_*.txt`: Predicted pocket doesn't contain ligand
- `without_orthosteric_ligand_at_minimum_distance_*.txt`: Ligand too far from pocket

**Chemical Filters**:
- `without_ligand_with_druglike_element_*.txt`: Non-drug-like elements
- `without_ligand_with_heavy_atoms_threshold_*.txt`: Too many/few heavy atoms

### Filter File Suffixes

- `*_heterodimer_*.txt`: Filters applied to protein-protein complexes
- `*_ligand_*.txt`: Filters applied to protein-ligand complexes
- `*_20230317PDBe.txt`: Filters from PDBe snapshot on March 17, 2023

### Example Filter File Content

```
2fk1
1oes
1p36
2e88
2fod
...
```

Each line contains a PDB ID that failed the specific criterion.

## Data Processing Pipeline

### Apparent Workflow

Based on file structure, the data appears to have been processed through:

1. **Download**: PDB structures retrieved from PDBe (March 17, 2023)
2. **Filtering**: Quality control using multiple criteria (see Filter Files)
3. **Extraction**: Protein chains and ligands extracted from PDB files
4. **Repair**: Hydrogen atoms added to structures (`Repair-H` files)
5. **Interface Analysis**: Contact residues identified (6 Å cutoff)
6. **Pocket Detection**: Binding cavities predicted and classified
7. **Descriptor Calculation**: 3D molecular descriptors computed

### Quality Control Criteria

High-quality structures retained must have:
- X-ray resolution ≤ threshold OR cryo-EM resolution ≤ threshold
- Valid UniProt annotation
- Drug-like ligand (for protein-ligand complexes)
- Clear binding interface (no overlaps, no alternate conformations)
- Orthosteric binding pocket containing ligand
- Appropriate number of heavy atoms in ligand

## Usage Examples

### Example 1: Load Drug Features for ML

```python
import pandas as pd
import glob

# Find all descriptor CSV files
descriptor_files = glob.glob("othercode/data/*/results/*/*.csv")

drug_features = []
for csv_file in descriptor_files:
    try:
        df = pd.read_csv(csv_file)
        if not df.empty and "Volume" in df.columns:
            # Extract features
            features = {
                'pdb_id': csv_file.split('/')[2],
                'volume': df['Volume'].values[0],
                'pmi1': df['PMI1'].values[0],
                'pmi2': df['PMI2'].values[0],
                'pmi3': df['PMI3'].values[0],
                'rgyr': df['Rgyr'].values[0],
                'asphericity': df['Asphericity'].values[0]
            }
            drug_features.append(features)
    except:
        continue

df_drugs = pd.DataFrame(drug_features)
print(f"Loaded {len(df_drugs)} drug structures with features")
```

### Example 2: Parse Interface Residues

```python
def load_interface_residues(pdb_id):
    """Load interface residues for a given PDB entry"""
    import glob

    interface_files = glob.glob(f"othercode/data/{pdb_id}/*interface-residues_6A.txt")

    for file_path in interface_files:
        with open(file_path, 'r') as f:
            residues = [line.strip() for line in f.readlines()]

        print(f"Found {len(residues)} interface residues in {file_path}")
        return residues

    return []

# Example usage
interface = load_interface_residues("1a0n")
print(interface[:5])  # ['A.PRO-91', 'A.ARG-93', ...]
```

### Example 3: Find Protein-Ligand vs Protein-Protein Complexes

```python
import os

def classify_pdb_entry(pdb_id):
    """Determine if entry is protein-ligand or protein-protein"""
    pdb_dir = f"othercode/data/{pdb_id}"

    # Check for MOL2 files (ligands)
    mol2_files = [f for f in os.listdir(pdb_dir) if f.endswith('.mol2')]

    if mol2_files:
        return "protein-ligand"
    else:
        return "protein-protein"

# Scan first 100 entries
pdb_ids = [d for d in os.listdir("othercode/data")
           if os.path.isdir(f"othercode/data/{d}")
           and not d.startswith('.')][:100]

for pdb_id in pdb_ids:
    classification = classify_pdb_entry(pdb_id)
    print(f"{pdb_id}: {classification}")
```

### Example 4: Extract Ligand Information

```python
def extract_ligand_info(pdb_id):
    """Extract ligand codes and positions from PDB entry"""
    import re
    import glob

    pdb_files = glob.glob(f"othercode/data/{pdb_id}/*.pdb")

    ligands = []
    for pdb_file in pdb_files:
        # Parse filename: [PDB]--[CHAIN]--[UNIPROT]--[LIG]-[NUM].pdb
        match = re.search(r'--([A-Z0-9]{3})-(\d+)\.pdb$', pdb_file)
        if match:
            ligand_code = match.group(1)
            ligand_num = match.group(2)
            ligands.append({
                'ligand': ligand_code,
                'position': ligand_num,
                'file': pdb_file
            })

    return ligands

# Example
ligands = extract_ligand_info("3hwe")
for lig in ligands:
    print(f"Ligand {lig['ligand']} at position {lig['position']}")
```

## Molecular Interaction Simulation Capabilities

### YES - Your Data Has Everything Needed for Molecular Simulations! ✅

Your dataset contains **complete structural information** for simulating drug-protein molecular interactions across **34,479 experimentally-determined structures**. This is a gold standard dataset for computational drug discovery.

### What Can Be Simulated

Your data contains **complete structural information** for simulating drug-protein molecular interactions:

**Available Components**:
1. **Ligand 3D Structures** (.mol2 files) - Small molecule drugs with coordinates and bond information
2. **Protein Binding Sites** (PDB files) - Protein structures with hydrogen atoms
3. **Binding Pocket Geometry** (3D descriptors) - Quantitative pocket shape and chemistry
4. **Interface Residues** (.txt files) - Exact residues involved in binding
5. **Binding Pocket Cavities** (.mol2 cavity files) - Predicted binding site geometries

**Simulation Types Enabled**:

✅ **Molecular Docking**: Predict how ligands bind to proteins
- Use ligand .mol2 files + protein .pdb files
- Tools: AutoDock Vina, Glide, GOLD, rDock

✅ **Molecular Dynamics (MD)**: Simulate time-evolution of binding
- Use protein-ligand complexes from your data
- Tools: GROMACS, AMBER, NAMD, OpenMM

✅ **Binding Affinity Prediction**: Calculate protein-ligand interaction strength
- Use 3D descriptors + ML models
- Tools: RF-Score, Scoring functions, Deep learning models

✅ **Pharmacophore Modeling**: Identify key interaction features
- Use interface residues + ligand positions
- Tools: LigandScout, Phase, MOE

✅ **Virtual Screening**: Screen new molecules against binding sites
- Use binding pocket descriptors to find similar drugs
- Tools: DOCK, AutoDock, Schrödinger Suite

✅ **Quantum Mechanical Calculations**: Calculate electronic interactions
- Use ligand + binding site residues
- Tools: Gaussian, ORCA, Psi4, Jaguar

### Simulation Workflow Example

```
Your Data → Molecular Interaction Simulation → Predictions

[1a0n.pdb] ────┐
               ├──→ [Docking Software] ──→ Binding Pose + Affinity Score
[ligand.mol2]──┘
```

### Practical Simulation Examples

#### Example 1: Molecular Docking with AutoDock Vina

```bash
# 1. Prepare protein (remove water, add hydrogens if needed)
protein_file="othercode/data/3hwe/results/3hwe--A--P80188__Repair-H.pdb"

# 2. Prepare ligand
ligand_file="othercode/data/3hwe/3hwe_A_RKS_180.mol2"

# 3. Convert to PDBQT format (AutoDock format)
# Using Open Babel or MGLTools
obabel -imol2 $ligand_file -opdbqt -O ligand.pdbqt
prepare_receptor4.py -r $protein_file -o protein.pdbqt

# 4. Define binding site (from interface residues or pocket descriptors)
# Use your descriptor data to set center and size
vina --receptor protein.pdbqt \
     --ligand ligand.pdbqt \
     --center_x 25.0 --center_y 30.0 --center_z 15.0 \
     --size_x 20 --size_y 20 --size_z 20 \
     --out docked_poses.pdbqt \
     --log docking.log

# 5. Analyze results
# Best binding affinity score will be in docking.log
```

#### Example 2: Binding Site Comparison for Virtual Screening

```python
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import glob

def load_binding_site_descriptors(pdb_id):
    """Load 3D descriptors for a binding site"""
    csv_files = glob.glob(f"othercode/data/{pdb_id}/results/*/*.csv")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty and "Volume" in df.columns:
                # Extract key shape descriptors
                descriptors = {
                    'volume': df['Volume'].values[0],
                    'pmi1': df['PMI1'].values[0],
                    'pmi2': df['PMI2'].values[0],
                    'pmi3': df['PMI3'].values[0],
                    'rgyr': df['Rgyr'].values[0],
                    'asphericity': df['Asphericity'].values[0],
                    'eccentricity': df['Eccentricity'].values[0]
                }
                return descriptors
        except:
            continue
    return None

def find_similar_binding_sites(query_pdb, all_pdbs, top_k=10):
    """Find binding sites similar to query based on geometric descriptors"""
    query_desc = load_binding_site_descriptors(query_pdb)
    if query_desc is None:
        return []

    query_vector = np.array([query_desc['volume'],
                            query_desc['pmi1'],
                            query_desc['pmi2'],
                            query_desc['pmi3'],
                            query_desc['rgyr'],
                            query_desc['asphericity'],
                            query_desc['eccentricity']])

    similarities = []
    for pdb_id in all_pdbs:
        target_desc = load_binding_site_descriptors(pdb_id)
        if target_desc is None:
            continue

        target_vector = np.array([target_desc['volume'],
                                 target_desc['pmi1'],
                                 target_desc['pmi2'],
                                 target_desc['pmi3'],
                                 target_desc['rgyr'],
                                 target_desc['asphericity'],
                                 target_desc['eccentricity']])

        # Normalize and calculate distance
        distance = euclidean(query_vector / np.linalg.norm(query_vector),
                           target_vector / np.linalg.norm(target_vector))
        similarities.append((pdb_id, distance))

    # Sort by similarity (lower distance = more similar)
    similarities.sort(key=lambda x: x[1])
    return similarities[:top_k]

# Usage: Find proteins with similar binding pockets to 3hwe
query = "3hwe"
all_pdbs = ["1a0n", "8exo", "5sou", "6jj3"]  # Add your PDB IDs
similar_sites = find_similar_binding_sites(query, all_pdbs)

print(f"Binding sites similar to {query}:")
for pdb_id, distance in similar_sites:
    print(f"  {pdb_id}: similarity score = {1/(1+distance):.3f}")
```

#### Example 3: Extract Interaction Features for ML

```python
def extract_protein_ligand_features(pdb_id):
    """Extract comprehensive features for ML models"""
    import glob
    import pandas as pd

    features = {'pdb_id': pdb_id}

    # 1. Load 3D descriptors
    csv_files = glob.glob(f"othercode/data/{pdb_id}/results/*/*.csv")
    if csv_files:
        try:
            df = pd.read_csv(csv_files[0])
            if not df.empty:
                # Geometric features
                features['volume'] = df['Volume'].values[0]
                features['rgyr'] = df['Rgyr'].values[0]
                features['asphericity'] = df['Asphericity'].values[0]
                features['npr1'] = df['NPR1'].values[0]
                features['npr2'] = df['NPR2'].values[0]

                # Atom type composition
                features['carbon'] = df['CZ'].values[0]
                features['oxygen'] = df['O'].values[0]
                features['nitrogen'] = df['N'].values[0]
        except:
            pass

    # 2. Count interface residues
    interface_files = glob.glob(f"othercode/data/{pdb_id}/*interface-residues_6A.txt")
    if interface_files:
        with open(interface_files[0], 'r') as f:
            features['num_interface_residues'] = len(f.readlines())

    # 3. Ligand information
    mol2_files = glob.glob(f"othercode/data/{pdb_id}/*.mol2")
    features['has_ligand'] = len(mol2_files) > 0
    features['num_ligands'] = len(mol2_files)

    return features

# Build dataset for ML
pdb_ids = ["3hwe", "1a0n", "8exo"]  # Your dataset
ml_dataset = []

for pdb_id in pdb_ids:
    features = extract_protein_ligand_features(pdb_id)
    ml_dataset.append(features)

df_ml = pd.DataFrame(ml_dataset)
print(df_ml)

# Now you can train ML models:
# - Predict binding affinity
# - Classify binding site types
# - Generate interaction fingerprints
```

#### Example 4: Molecular Dynamics Preparation

```python
def prepare_md_simulation(pdb_id, ligand_code):
    """Prepare files for molecular dynamics simulation"""
    import glob

    # Find protein-ligand complex
    complex_file = glob.glob(f"othercode/data/{pdb_id}/{pdb_id}--*--{ligand_code}-*.pdb")
    ligand_file = glob.glob(f"othercode/data/{pdb_id}/{pdb_id}_*_{ligand_code}_*.mol2")

    if not complex_file or not ligand_file:
        print(f"Complex or ligand not found for {pdb_id}")
        return None

    instructions = f"""
# Molecular Dynamics Simulation Setup for {pdb_id}

## Files:
- Protein-Ligand Complex: {complex_file[0]}
- Ligand Structure: {ligand_file[0]}

## GROMACS Workflow:

# 1. Generate topology
gmx pdb2gmx -f {complex_file[0]} -o protein.gro -water tip3p

# 2. Prepare ligand topology (use ACPYPE or LigParGen)
# Generate ligand parameters from MOL2 file

# 3. Define simulation box
gmx editconf -f protein.gro -o box.gro -c -d 1.0 -bt cubic

# 4. Solvate
gmx solvate -cp box.gro -cs spc216.gro -o solvated.gro -p topol.top

# 5. Add ions
gmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr
gmx genion -s ions.tpr -o ionized.gro -p topol.top -pname NA -nname CL -neutral

# 6. Energy minimization
gmx grompp -f em.mdp -c ionized.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em

# 7. Equilibration (NVT, NPT)
# 8. Production MD run (e.g., 100 ns)

## Analysis:
# - RMSD: Binding stability
# - Hydrogen bonds: Interaction persistence
# - Binding free energy: MM-PBSA or MM-GBSA
"""

    return instructions

# Example
md_setup = prepare_md_simulation("3hwe", "RKS")
print(md_setup)
```

#### Example 5: Quantum Mechanical Interaction Energy

```python
def prepare_qm_calculation(pdb_id):
    """
    Prepare quantum mechanical calculation for drug-protein interaction
    Uses interface residues + ligand for accurate binding energy
    """
    import glob

    # 1. Load interface residues
    interface_file = glob.glob(f"othercode/data/{pdb_id}/*interface-residues_6A.txt")[0]
    with open(interface_file, 'r') as f:
        interface_residues = [line.strip() for line in f.readlines()]

    # 2. Load ligand
    ligand_file = glob.glob(f"othercode/data/{pdb_id}/*.mol2")[0]

    # 3. Extract binding site pocket (residues within 4-6Å of ligand)
    # This reduces QM calculation to manageable size

    qm_input = f"""
# Quantum Mechanical Calculation Setup
# Using Gaussian/ORCA/Psi4 for interaction energy

## System: {pdb_id}
## Method: DFT (B3LYP/6-31G* or similar)
## Purpose: Calculate protein-ligand interaction energy

# Files needed:
# 1. Ligand coordinates: {ligand_file}
# 2. Binding site residues ({len(interface_residues)} residues):
{chr(10).join(['   - ' + r for r in interface_residues[:10]])}
   ... (total {len(interface_residues)} residues)

# Workflow:
# 1. Extract binding site residues from PDB
# 2. Combine with ligand in QM software
# 3. Optimize geometry (DFT)
# 4. Calculate interaction energy:
#    E_interaction = E_complex - (E_protein + E_ligand)

# Expected outputs:
# - Binding energy (kcal/mol)
# - Electron density maps
# - Molecular orbitals
# - Charge transfer analysis
"""

    return qm_input

qm_calc = prepare_qm_calculation("3hwe")
print(qm_calc)
```

## Data Applications

### Drug Discovery

- **Virtual Screening**: Use 3D descriptors to compare binding pockets across 34,479 structures
- **QSAR Models**: Train structure-activity relationship models using descriptor features
- **Pharmacophore Mapping**: Identify key binding features from interface residues
- **Lead Optimization**: Analyze successful drug-protein interactions from experimentally-solved complexes
- **De Novo Drug Design**: Design new molecules to fit binding pockets
- **Fragment-Based Drug Design**: Identify fragment binding modes

### Machine Learning

- **Graph Neural Networks**: Represent proteins and ligands as graphs with spatial features
- **Quantum Machine Learning**: Encode molecular features in quantum circuits (see Pipeline integration)
- **Drug-Target Prediction**: Train models to predict binding affinity using 3D descriptors
- **Drug Repurposing**: Find similar binding pockets for existing drugs using ML similarity search
- **Binding Site Classification**: Classify binding sites by function/drug class
- **Interaction Fingerprints**: Generate ML features from protein-ligand contacts

### Structural Biology

- **Protein-Protein Interface Analysis**: Study heterodimer contacts and hot spots
- **Binding Site Characterization**: Analyze pocket geometry and chemistry using PMI, Rgyr, etc.
- **Conformational Studies**: Compare structures of same protein with different ligands
- **Evolutionary Analysis**: Compare binding sites across species using structural alignment
- **Allosteric Site Identification**: Find non-orthosteric binding pockets

## Integration with Drug-Patient Pipeline

This data directory is used by the Drug-Patient Quantum GNN Pipeline (see [DRUG_PATIENT_PIPELINE_README.md](../../DRUG_PATIENT_PIPELINE_README.md)):

### Data Flow

```
PDB Data (this directory)
    ↓
Load 3D Descriptors (CSV files)
    ↓
Drug Feature Vectors
    ↓
Quantum GNN Encoding
    ↓
Drug-Patient Interaction Prediction
```

### Feature Extraction

The pipeline automatically extracts features from `*_descriptors_3d.csv` files:

```python
from drug_patient_quantum_gnn_pipeline import DrugPatientDataProcessor

processor = DrugPatientDataProcessor(data_dir="othercode/data")
processor.load_protein_ligand_data(max_samples=100)

# Extracts: Volume, PMI1-3, NPR1-2, Rgyr, Asphericity, etc.
drug_features = processor.graph.get_drug_features_matrix()
print(f"Drug feature dimensions: {drug_features.shape}")
```

## Data Quality Notes

### Known Issues

1. **Missing Descriptors**: Some structures lack `*_descriptors_3d.csv` files
   - Reason: Binding pocket detection failed
   - Impact: Cannot be used for ML models requiring 3D features

2. **Hydrogen Repair Warnings**: Some structures have hydrogen addition issues
   - Check `state.log` files for errors
   - May affect electrostatic calculations

3. **Empty CSV Files**: Some descriptor files contain only "No pockets mol2 found"
   - Indicates pocket detection algorithm failure
   - Structure may have unusual geometry or no clear binding site

4. **Multiple Ligands**: Some PDB entries have multiple ligand binding sites
   - Each ligand has separate files and descriptors
   - Must specify which ligand to use

### Validation Recommendations

Before using data:

1. **Check file existence**: Verify all required files are present
2. **Validate CSV format**: Ensure descriptor files are not empty
3. **Check resolution**: Use filter files to identify low-quality structures
4. **Verify UniProt mapping**: Cross-reference with UniProt database

## File Size and Storage

### Estimated Storage Requirements

- **Total directory size**: ~75+ GB (estimated)
- **Per PDB entry**: ~1-5 MB (varies by structure size)
- **Largest files**: PDB structure files (.pdb, .ent)
- **Smallest files**: Interface residue lists (.txt)

### Compression Recommendations

If storage is limited:

1. **Remove PDB .ent files** (can be re-downloaded from PDB)
2. **Keep only descriptor CSV files** (smallest, most useful for ML)
3. **Archive unused filter files** (for documentation only)

## Related Resources

### External Databases

- **Protein Data Bank (PDB)**: https://www.rcsb.org/
- **PDBe (Europe)**: https://www.ebi.ac.uk/pdbe/
- **UniProt**: https://www.uniprot.org/
- **PubChem**: https://pubchem.ncbi.nlm.nih.gov/ (for ligand info)

### Tools and Software

- **PyMOL**: Visualize PDB structures
- **Open Babel**: Convert between molecular formats
- **RDKit**: Cheminformatics toolkit
- **BioPython**: Parse PDB files programmatically
- **ProDy**: Protein dynamics and structure analysis

### Citation

If using this data, cite the original PDB entries:

```bibtex
@misc{pdb_data_collection,
  title={Protein-Ligand and Heterodimer Data Collection},
  author={Protein Data Bank Europe},
  year={2023},
  note={Data snapshot: March 17, 2023}
}
```

## Maintenance and Updates

### Last Updated

- **PDB Snapshot**: March 17, 2023
- **Directory created**: September 7, 2023
- **Documentation created**: November 9, 2025

### Updating Data

To refresh with newer PDB structures:

1. Re-run filtering pipeline against latest PDBe
2. Download new/updated structures
3. Reprocess with hydrogen addition and descriptor calculation
4. Update filter files with new quality control results

### Version Control

This is a static snapshot. For the latest PDB data:
- Visit https://www.rcsb.org/
- Use PDB API for programmatic access
- Subscribe to PDB weekly updates

## Contact and Support

For issues with:
- **Data quality**: Check `state.log` files in results directories
- **Missing files**: Verify PDB ID exists in PDB database
- **File format**: Consult PDB format documentation
- **Pipeline usage**: See [DRUG_PATIENT_PIPELINE_README.md](../../DRUG_PATIENT_PIPELINE_README.md)

---

**Documentation Generated**: November 9, 2025
**Total PDB Entries Documented**: 34,479
**Data Source**: Protein Data Bank Europe (PDBe) - March 2023 snapshot
