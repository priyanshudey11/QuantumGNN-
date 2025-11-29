
import json
import sys
import os

NOTEBOOK_PATH = "/home/priyanshu/QuantumGNN-/Actuall/train_model_spark.ipynb"

def fix_notebook_imports():
    try:
        with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        cells = nb.get('cells', [])
        updated = False
        
        for cell in cells:
            if cell.get('cell_type') == 'code':
                source = "".join(cell.get('source', []))
                # Identify the cell by looking for the autoreload magic or our previous injection
                if "%load_ext autoreload" in source:
                    print("Found import cell.")
                    
                    current_source = cell.get('source', [])
                    
                    # Find where the imports start (after our injection or at the top)
                    # We'll just look for %load_ext autoreload and keep everything from there
                    split_idx = -1
                    for i, line in enumerate(current_source):
                        if "%load_ext autoreload" in line:
                            split_idx = i
                            break
                    
                    if split_idx != -1:
                        rest_of_cell = current_source[split_idx:]
                        
                        # New robust reload block
                        new_header = [
                            "import importlib\n",
                            "import drug_patient_qgnn.data_processing\n",
                            "import drug_patient_qgnn\n",
                            "\n",
                            "# Reload module first to get new class definition\n",
                            "importlib.reload(drug_patient_qgnn.data_processing)\n",
                            "# Reload package to update __init__ imports\n",
                            "importlib.reload(drug_patient_qgnn)\n",
                            "\n",
                            "print('Force reloaded DrugProteinDataProcessor and package')\n",
                            "\n"
                        ]
                        
                        cell['source'] = new_header + rest_of_cell
                        updated = True
                        break
        
        if updated:
            with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print(f"Successfully updated {NOTEBOOK_PATH} with robust reload logic")
        else:
            print("Could not find the import cell to update.")
            
    except Exception as e:
        print(f"Error updating notebook: {e}")
        sys.exit(1)

if __name__ == "__main__":
    fix_notebook_imports()
