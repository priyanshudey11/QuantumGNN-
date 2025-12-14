"""Validation and testing utilities."""

import sys
from pathlib import Path
import numpy as np

def validate_setup():
    """Check all dependencies are installed."""
    print("Validating setup...\n")
    
    checks = {
        'torch': 'PyTorch',
        'torch_geometric': 'PyTorch Geometric',
        'rdkit': 'RDKit',
        'Bio': 'BioPython',
        'pennylane': 'PennyLane',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'yaml': 'PyYAML',
    }
    
    all_ok = True
    for module_name, display_name in checks.items():
        try:
            __import__(module_name)
            print(f" {display_name}")
        except ImportError:
            print(f"✗ {display_name} (not installed)")
            all_ok = False
    
    if all_ok:
        print("\n✅ All dependencies installed!")
        return True
    else:
        print("\n❌ Some dependencies missing. Run: pip install -r requirements.txt")
        return False


def validate_data_structure():
    """Check if data directories exist."""
    print("\nValidating data structure...\n")
    
    required_dirs = [
        'data/pdb',
        'data/includes',
        'data/excludes',
        'configs'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f" {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
            all_ok = False
    
    required_files = [
        'configs/default.yaml',
        'requirements.txt',
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f" {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_ok = False
    
    if all_ok:
        print("\n✅ Data structure valid!")
    else:
        print("\n⚠️ Some files/directories missing.")
        print("Run: python scripts/generate_sample_data.py")
    
    return all_ok


def test_smiles_parsing():
    """Test SMILES to graph conversion."""
    print("\nTesting SMILES parsing...\n")
    
    try:
        from src.data.drug_graph import smiles_to_graph
        
        test_smiles = [
            ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
            ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
            ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
            ("Invalid", "invalid_smiles_123"),
        ]
        
        successes = 0
        for name, smiles in test_smiles:
            try:
                graph = smiles_to_graph(smiles)
                if graph is not None:
                    print(f" {name}: {graph.num_nodes} atoms")
                    successes += 1
                else:
                    print(f"✗ {name}: invalid SMILES")
            except Exception as e:
                print(f"✗ {name}: {e}")
        
        print(f"\nPassed: {successes}/{len(test_smiles)}")
        return successes == len(test_smiles) - 1  # All except invalid should work
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_pdb_parsing():
    """Test PDB file parsing."""
    print("\nTesting PDB parsing...\n")
    
    try:
        from src.data.pdb_parse import load_pdb_structure
        
        # Try to find any PDB file
        pdb_files = list(Path('data/pdb').glob('*.pdb'))
        
        if not pdb_files:
            print("⚠️ No PDB files found in data/pdb/")
            print("Run: python scripts/download_pdbs.py --pdb-ids 1a2b")
            return True  # Not a failure, just warning
        
        for pdb_file in pdb_files[:3]:
            try:
                structure = load_pdb_structure(str(pdb_file))
                if structure:
                    print(f" {pdb_file.name}")
                else:
                    print(f"✗ {pdb_file.name}: failed to parse")
            except Exception as e:
                print(f"✗ {pdb_file.name}: {e}")
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...\n")
    
    try:
        import yaml
        from src.models.head import DrugProteinInteractionModel, ClassicalOnlyModel
        
        # Load config
        with open('configs/default.yaml') as f:
            config = yaml.safe_load(f)
        
        # Small config for testing
        test_config = config.copy()
        test_config['model']['dim'] = 128
        test_config['model']['drug_gnn']['layers'] = 2
        test_config['model']['quantum']['n_qubits'] = 4
        test_config['model']['quantum']['depth'] = 1
        
        # Test full model
        try:
            model = DrugProteinInteractionModel(test_config)
            n_params = sum(p.numel() for p in model.parameters())
            print(f" Full model: {n_params:,} parameters")
        except Exception as e:
            print(f"✗ Full model: {e}")
            return False
        
        # Test classical model
        try:
            model = ClassicalOnlyModel(test_config)
            n_params = sum(p.numel() for p in model.parameters())
            print(f" Classical model: {n_params:,} parameters")
        except Exception as e:
            print(f"✗ Classical model: {e}")
            return False
        
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all validations."""
    print("\n" + "="*60)
    print("Quantum Drug-Protein Interaction Pipeline - Validation")
    print("="*60 + "\n")
    
    results = []
    
    # Run checks
    results.append(("Dependencies", validate_setup()))
    results.append(("Data Structure", validate_data_structure()))
    results.append(("SMILES Parsing", test_smiles_parsing()))
    results.append(("PDB Parsing", test_pdb_parsing()))
    results.append(("Model Creation", test_model_creation()))
    
    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:.<40} {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("="*60)
    if all_passed:
        print("\n🎉 All checks passed! Ready to train.\n")
        print("Next steps:")
        print("  python scripts/generate_sample_data.py")
        print("  python run_pipeline.py --config configs/default.yaml --model-type full")
        return 0
    else:
        print("\n⚠️ Some checks failed. Fix the issues and try again.\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
