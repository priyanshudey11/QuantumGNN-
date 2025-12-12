#!/usr/bin/env python3
"""
Integration test for alternate data source.
Verifies all components work end-to-end.
"""

import sys
from pathlib import Path

def test_loader():
    """Test AlternateDataLoader."""
    print("\n" + "="*60)
    print("TEST 1: AlternateDataLoader")
    print("="*60)
    
    from src.data.alternate_loader import AlternateDataLoader
    
    data_path = "/Users/priyanshudey/Code/Qunatum/othercode/data"
    loader = AlternateDataLoader(data_path)
    
    print(f"✓ Loaded {len(loader.complexes)} complexes")
    
    # Test getters
    liganded = loader.get_liganded_complexes()
    print(f"✓ Liganded: {len(liganded)}")
    
    protein_only = loader.get_protein_only_complexes()
    print(f"✓ Protein-only: {len(protein_only)}")
    
    # Test queries
    pdb_complexes = loader.get_by_pdb_id('4hvb')
    print(f"✓ Query by PDB (4hvb): {len(pdb_complexes)} complexes")
    
    # Test validation
    issues = loader.validate_complexes()
    print(f"✓ Validation: {sum(len(v) for v in issues.values())} issues found")
    
    return loader


def test_adapter(loader):
    """Test AlternateDataAdapter."""
    print("\n" + "="*60)
    print("TEST 2: AlternateDataAdapter")
    print("="*60)
    
    from src.data.data_adapter import AlternateDataAdapter
    
    adapter = AlternateDataAdapter(loader)
    
    # Test dataframe generation
    df = adapter.generate_pairs_dataframe()
    print(f"✓ Generated pairs DataFrame: {len(df)} rows")
    
    # Test drug lookup
    drug_lookup = adapter.create_drug_lookup()
    print(f"✓ Created drug lookup: {len(drug_lookup)} unique ligands")
    
    # Test protein lookup
    protein_lookup = adapter.create_protein_lookup()
    print(f"✓ Created protein lookup: {len(protein_lookup)} unique proteins")
    
    return adapter


def test_enhanced_loader():
    """Test enhanced loader with config."""
    print("\n" + "="*60)
    print("TEST 3: EnhancedLoader with Config")
    print("="*60)
    
    import yaml
    from src.data.enhanced_loader import load_data_with_alternate_support
    
    # Load test config
    with open('configs/test_alternate.yaml') as f:
        config = yaml.safe_load(f)
    
    # Load data
    pdb_metadata, drug_smiles, ligand_lookup = load_data_with_alternate_support(config)
    
    print(f"✓ Loaded {len(pdb_metadata)} PDB structures")
    print(f"✓ Loaded {len(drug_smiles)} drugs")
    print(f"✓ Ligand lookup keys: {len(ligand_lookup)}")
    
    return config, pdb_metadata, drug_smiles


def test_csv_export(adapter):
    """Test CSV export functionality."""
    print("\n" + "="*60)
    print("TEST 4: CSV Export")
    print("="*60)
    
    import os
    
    output_dir = Path("outputs/test_exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export
    adapter.save_pairs_csv(str(output_dir / "complexes.csv"))
    adapter.save_drug_lookup_csv(str(output_dir / "drug_lookup.csv"))
    adapter.save_protein_lookup_csv(str(output_dir / "protein_lookup.csv"))
    
    # Verify files exist and have content
    for fname in ["complexes.csv", "drug_lookup.csv", "protein_lookup.csv"]:
        fpath = output_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            lines = len(open(fpath).readlines())
            print(f"✓ {fname}: {size_kb:.1f} KB, {lines} lines")
        else:
            print(f"✗ {fname}: NOT FOUND")
            return False
    
    return True


def test_dataset_creation(config, pdb_metadata, drug_smiles):
    """Test dataset creation."""
    print("\n" + "="*60)
    print("TEST 5: Dataset Creation")
    print("="*60)
    
    from src.data.datasets import DrugProteinDataset
    
    # Create dataset
    dataset = DrugProteinDataset(
        drug_smiles,
        list(pdb_metadata.keys()),
        pdb_metadata,
        negatives_per_positive=config['sampling']['negatives_per_positive']
    )
    
    print(f"✓ Created dataset with {len(dataset.pairs)} pairs")
    
    # Test split
    train_pairs, val_pairs, test_pairs = dataset.split(
        val_split=config['sampling']['val_split'],
        test_split=config['sampling']['test_split'],
        seed=config['sampling']['scaffold_seed']
    )
    
    print(f"✓ Train split: {len(train_pairs)} pairs")
    print(f"✓ Val split: {len(val_pairs)} pairs")
    print(f"✓ Test split: {len(test_pairs)} pairs")
    
    return True


def test_cli_commands():
    """Test CLI commands."""
    print("\n" + "="*60)
    print("TEST 6: CLI Commands")
    print("="*60)
    
    import subprocess
    
    data_path = "/Users/priyanshudey/Code/Qunatum/othercode/data"
    
    # Test query
    result = subprocess.run(
        ["python", "integrate_data.py", data_path, "--pdb", "4hvb"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ CLI query command works")
    else:
        print(f"✗ CLI query failed: {result.stderr}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ALTERNATE DATA SOURCE INTEGRATION TEST SUITE")
    print("="*70)
    
    try:
        # Test 1: Loader
        loader = test_loader()
        
        # Test 2: Adapter
        adapter = test_adapter(loader)
        
        # Test 3: Enhanced loader
        config, pdb_metadata, drug_smiles = test_enhanced_loader()
        
        # Test 4: CSV Export
        if not test_csv_export(adapter):
            return 1
        
        # Test 5: Dataset creation
        if not test_dataset_creation(config, pdb_metadata, drug_smiles):
            return 1
        
        # Test 6: CLI
        if not test_cli_commands():
            return 1
        
        # Summary
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nNext steps:")
        print("1. Review exports in outputs/test_exports/")
        print("2. Update configs/default.yaml with your data path")
        print("3. Run: python run_pipeline.py --config configs/test_alternate.yaml")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
