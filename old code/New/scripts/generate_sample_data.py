"""Generate sample data for testing."""

import os
from pathlib import Path
import pandas as pd
import numpy as np


def create_sample_drugs_csv(output_path: str = "data/drugs.csv", n_drugs: int = 50):
    """Create sample drug SMILES data."""
    
    # Common drug SMILES patterns (simplified)
    sample_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "CN1CCC23C4C1CC5=C2C(=C(C(=C5)O)O)OC3C(C4)O",  # Morphine
        "Cc1c(nc(n1)Nc2ccc(cc2)S(=O)(=O)N)NS(=O)(=O)c3ccc(cc3)N",  # Sulfamethoxazole
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",  # Long chain
        "c1ccccc1",  # Benzene
        "CC(C)CC(C)(C)O",  # Tert-butyl alcohol
        "c1ccc2c(c1)ccc3c2cccc3",  # Anthracene
        "CC(=O)c1ccccc1O",  # Acetylsalicylic
    ]
    
    # Create dataset
    drugs_data = []
    for i in range(n_drugs):
        drug_id = f"drug_{i+1:04d}"
        smiles = sample_smiles[i % len(sample_smiles)]
        # Add slight variation for diversity
        if i % 3 == 0 and i > 0:
            smiles = smiles + "C"  # Add carbon
        drugs_data.append({"drug_id": drug_id, "smiles": smiles})
    
    df = pd.DataFrame(drugs_data)
    
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(df)} drugs")
    return output_path


def create_sample_pdb_lists(
    output_dir: str = "data/includes",
    n_proteins_orthosteric: int = 20,
    n_proteins_allosteric: int = 15
):
    """Create sample PDB ID include lists."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate PDB IDs
    pdb_ids_ortho = [f"{np.random.randint(1000,9999)}{chr(np.random.randint(65,91))}" 
                     for _ in range(n_proteins_orthosteric)]
    pdb_ids_allosteric = [f"{np.random.randint(1000,9999)}{chr(np.random.randint(65,91))}" 
                          for _ in range(n_proteins_allosteric)]
    
    # Save orthosteric
    ortho_df = pd.DataFrame({"pdb_id": pdb_ids_ortho})
    ortho_path = Path(output_dir) / "HD-PL_part1_matrix_liganded_orthosteric.csv"
    ortho_df.to_csv(ortho_path, index=False)
    print(f"Created {ortho_path} with {len(ortho_df)} PDB IDs")
    
    # Save allosteric
    allosteric_df = pd.DataFrame({"pdb_id": pdb_ids_allosteric})
    allosteric_path = Path(output_dir) / "HD-PL_part2_matrix_liganded_allosteric.csv"
    allosteric_df.to_csv(allosteric_path, index=False)
    print(f"Created {allosteric_path} with {len(allosteric_df)} PDB IDs")


def create_sample_exclusion_lists(output_dir: str = "data/excludes", n_excludes: int = 5):
    """Create sample exclusion lists."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create some exclusion lists
    exclusion_categories = [
        "without_low_resolution",
        "without_uniprot_annotation",
        "without_secondary_structure",
        "without_ligand_coordinates",
        "without_metal_ions"
    ]
    
    for category in exclusion_categories[:n_excludes]:
        pdb_ids = [f"{np.random.randint(1000,9999)}{chr(np.random.randint(65,91))}" 
                   for _ in range(np.random.randint(2, 8))]
        
        exclude_path = Path(output_dir) / f"{category}.txt"
        with open(exclude_path, 'w') as f:
            for pdb_id in pdb_ids:
                f.write(f"{pdb_id}\n")
        
        print(f"Created {exclude_path} with {len(pdb_ids)} PDB IDs")


def create_sample_ligand_lookup(output_path: str = "data/ligand_lookup.csv", n_ligands: int = 20):
    """Create sample ligand lookup table."""
    
    sample_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    ]
    
    ligands = []
    for i in range(n_ligands):
        # 3-letter PDB ligand codes
        lig_3let = f"{chr(65 + (i % 26))}{chr(66 + (i % 26))}{chr(67 + (i % 26))}"
        smiles = sample_smiles[i % len(sample_smiles)]
        ligands.append({"pdb_lig_3let": lig_3let, "smiles": smiles})
    
    df = pd.DataFrame(ligands)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(df)} ligands")


if __name__ == "__main__":
    print("Generating sample data...\n")
    
    create_sample_drugs_csv(n_drugs=50)
    create_sample_pdb_lists()
    create_sample_exclusion_lists()
    create_sample_ligand_lookup()
    
    print("\n✅ Sample data generation complete!")
    print("\nNext steps:")
    print("1. Download actual PDB files: data/pdb/{pdbid}.pdb")
    print("2. Update configs/default.yaml with your data paths")
    print("3. Run: python -m src.train --config configs/default.yaml")
