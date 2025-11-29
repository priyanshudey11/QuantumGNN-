
import json
import sys

NOTEBOOK_PATH = "/home/priyanshu/QuantumGNN-/Actuall/train_model_spark.ipynb"

def inject_diagnostics():
    try:
        with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        cells = nb.get('cells', [])
        updated = False
        
        for cell in cells:
            if cell.get('cell_type') == 'code':
                source = "".join(cell.get('source', []))
                if "from drug_patient_qgnn import" in source:
                    print("Found import cell. Injecting diagnostics...")
                    
                    current_source = cell.get('source', [])
                    
                    # Add diagnostic print
                    new_source = current_source + [
                        "\n",
                        "import drug_patient_qgnn\n",
                        "print(f'Using drug_patient_qgnn from: {drug_patient_qgnn.__file__}')\n"
                    ]
                    
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
    inject_diagnostics()
