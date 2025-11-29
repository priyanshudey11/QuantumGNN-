import os
import glob
import pandas as pd
import sys

def debug_data_loading(data_dir):
    print(f" Inspecting data directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        return

    # Expected columns from data_processing.py
    expected_descriptor_cols = [
        'Volume', 'PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2',
        'Rgyr', 'Asphericity', 'SpherocityIndex', 'Eccentricity',
        'InertialShapeFactor'
    ]
    expected_atom_cols = ['CZ', 'CA', 'O', 'OD1', 'OG', 'N', 'NZ', 'DU']
    
    print("\nExpected Columns:")
    print(f"  Descriptors: {expected_descriptor_cols}")
    print(f"  Atom Types (Optional): {expected_atom_cols}")

    # Search for CSV files
    pattern = os.path.join(data_dir, "**", "results", "**", "*_descriptors_3d.csv")
    print(f"\n Searching for files matching: {pattern}")
    csv_files = glob.glob(pattern, recursive=True)
    
    # Filter hidden files
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith('.')]

    if not csv_files:
        print(" No descriptor CSV files found.")
        return

    print(f"Found {len(csv_files)} files.")
    
    # Inspect the first few failing files
    print("\n🧪 Analyzing first 5 files...")
    
    issues_found = 0
    for i, csv_file in enumerate(csv_files[:5]):
        print(f"\n[{i+1}] Checking: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            columns = df.columns.tolist()
            print(f"   Found columns: {columns}")
            
            # Check for matches
            found_descriptors = [col for col in expected_descriptor_cols if col in columns]
            found_atoms = [col for col in expected_atom_cols if col in columns]
            
            missing_descriptors = [col for col in expected_descriptor_cols if col not in columns]
            
            if found_descriptors:
                print(f"   Valid descriptors found: {len(found_descriptors)}/{len(expected_descriptor_cols)}")
            else:
                print(f"   NO valid descriptors found!")
                issues_found += 1
                
            if missing_descriptors:
                print(f"    Missing descriptors: {missing_descriptors}")
                
        except Exception as e:
            print(f"    Error reading file: {e}")
            issues_found += 1

    print("\n" + "="*50)
    if issues_found > 0:
        print("DIAGNOSIS: The CSV files do not contain the expected column names.")
        print("   Please check if your data generation process used different column names.")
        print("   Common mismatches: lowercase vs uppercase (e.g., 'volume' vs 'Volume').")
    else:
        print("DIAGNOSIS: The first 5 files look okay. The error might be in other files.")
    print("="*50)

if __name__ == "__main__":
    # Default path from the user's error message
    DEFAULT_DATA_DIR = "/media/priyanshu/SD/othercode/data"
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = DEFAULT_DATA_DIR
        
    debug_data_loading(data_dir)
