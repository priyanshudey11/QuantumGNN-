import sys
import os

# Add the current directory to path so we can import the module
sys.path.append(os.getcwd())

from Actuall.drug_patient_qgnn.data_processing import DrugProteinDataProcessor

def verify_fix():
    print("🧪 Testing Data Loading Fix...")
    
    # Initialize processor
    processor = DrugProteinDataProcessor()
    
    # Load a small sample of data (enough to likely hit some bad files)
    # We know there are ~15k files and ~2k are bad, so 500 samples should hit some.
    print("   Loading sample of 500 pockets...")
    try:
        count = processor.load_protein_ligand_data(max_samples=500)
        print(f"✅ Successfully loaded {count} pockets.")
    except Exception as e:
        print(f"❌ Failed with error: {e}")
        sys.exit(1)

    print("\n✨ Verification Complete!")

if __name__ == "__main__":
    verify_fix()
