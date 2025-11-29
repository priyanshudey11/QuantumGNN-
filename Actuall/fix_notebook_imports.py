
import json
import sys

NOTEBOOK_PATH = "/home/priyanshu/QuantumGNN-/Actuall/train_model_spark.ipynb"

def fix_imports():
    try:
        with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        cells = nb.get('cells', [])
        updated = False
        
        for cell in cells:
            if cell.get('cell_type') == 'code':
                source = "".join(cell.get('source', []))
                if "import os" in source and "import sys" in source:
                    print("Found import cell. Injecting autoreload...")
                    
                    current_source = cell.get('source', [])
                    
                    # Check if autoreload is already there
                    if any("%load_ext autoreload" in line for line in current_source):
                        print("Autoreload already present.")
                        return

                    new_source = [
                        "%load_ext autoreload\n",
                        "%autoreload 2\n",
                        "\n"
                    ] + current_source
                    
                    cell['source'] = new_source
                    updated = True
                    break
        
        if updated:
            with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print(f"Successfully updated {NOTEBOOK_PATH}")
        else:
            print("Could not find the import cell to update.")
            
    except Exception as e:
        print(f"Error updating notebook: {e}")
        sys.exit(1)

if __name__ == "__main__":
    fix_imports()
