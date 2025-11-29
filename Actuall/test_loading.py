
import sys
import os
sys.path.append(os.path.abspath("Actuall"))
from drug_patient_qgnn.data_processing import DrugProteinDataProcessor

def test_loading():
    print("Testing data loading...")
    processor = DrugProteinDataProcessor(data_dir="/media/priyanshu/SD/othercode/data")
    
    print("Calling load_drug_data_from_pdb...")
    n_drugs = processor.load_drug_data_from_pdb(max_samples=100)
    print(f"Loaded {n_drugs} drugs")
    
    print("Calling create_synthetic_patient_data...")
    processor.create_synthetic_patient_data(n_patients=10)
    print(f"Created {processor.graph.num_pockets()} patients")
    
    print("Stats:", processor.get_statistics())

if __name__ == "__main__":
    test_loading()
