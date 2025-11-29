
import json
import sys

NOTEBOOK_PATH = "/home/priyanshu/QuantumGNN-/Actuall/train_model_spark.ipynb"

def force_reload_in_notebook():
    try:
        with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        cells = nb.get('cells', [])
        updated = False
        
        for cell in cells:
            if cell.get('cell_type') == 'code':
                source = "".join(cell.get('source', []))
                if "from drug_patient_qgnn import" in source:
                    print("Found import cell. Injecting importlib.reload...")
                    
                    current_source = cell.get('source', [])
                    
                    # Construct the new source with explicit reload
                    new_source = [
                        "import importlib\n",
                        "import drug_patient_qgnn.data_processing\n",
                        "importlib.reload(drug_patient_qgnn.data_processing)\n",
                        "# Update the alias to point to the reloaded class\n",
                        "DrugPatientDataProcessor = drug_patient_qgnn.data_processing.DrugProteinDataProcessor\n",
                        "# Also import the new name\n",
                        "from drug_patient_qgnn.data_processing import DrugProteinDataProcessor\n",
                        "print('Force reloaded DrugProteinDataProcessor and updated alias')\n",
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
    force_reload_in_notebook()
