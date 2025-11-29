
import sys
import os
sys.path.append('/home/priyanshu/QuantumGNN-/Actuall')

try:
    from drug_patient_qgnn.data_processing import DrugProteinDataProcessor
    print(f"Module file: {sys.modules['drug_patient_qgnn.data_processing'].__file__}")
    
    processor = DrugProteinDataProcessor()
    if hasattr(processor, 'load_real_data'):
        print("SUCCESS: load_real_data method exists.")
    else:
        print("FAILURE: load_real_data method does NOT exist.")
        print("Available attributes:", dir(processor))
        
except Exception as e:
    print(f"Error: {e}")
