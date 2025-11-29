
import sys
import os
import torch
from drug_patient_qgnn.data_processing import DrugPatientDataProcessor

# Data path
DATA_DIR = "/media/priyanshu/SD/othercode/data"

def test_loading():
    print("Testing Real Data Loading...")
    
    # Initialize processor
    processor = DrugPatientDataProcessor(data_dir=DATA_DIR)
    
    # Load a small subset of data
    print(f"Loading data from: {DATA_DIR}")
    counts = processor.load_real_data(data_dir=DATA_DIR, max_samples=10)
    
    print("\nLoading Results:")
    print(f"Proteins: {counts['proteins']}")
    print(f"Drugs: {counts['drugs']}")
    print(f"Interactions: {counts['interactions']}")
    
    # Verify graph structure
    print("\nGraph Statistics:")
    stats = processor.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
        
    # Check if we have interactions
    if counts['interactions'] > 0:
        print("\nSUCCESS: Interactions loaded successfully.")
    else:
        print("\nFAILURE: No interactions loaded.")

if __name__ == "__main__":
    test_loading()
