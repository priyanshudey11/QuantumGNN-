
import json
import sys

NOTEBOOK_PATH = "/home/priyanshu/QuantumGNN-/Actuall/train_model_spark.ipynb"

def update_notebook():
    try:
        with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        cells = nb.get('cells', [])
        updated = False
        
        for cell in cells:
            if cell.get('cell_type') == 'code':
                source = "".join(cell.get('source', []))
                # Identify the data loading cell by looking for the processor initialization and old method calls
                if "DrugPatientDataProcessor" in source and "load_drug_data_from_pdb" in source:
                    print("Found data loading cell. Updating...")
                    
                    new_source = [
                        "print(\"Loading Data with Spark\")\n",
                        "\n",
                        "# Initialize processor\n",
                        "processor = DrugPatientDataProcessor(data_dir=DATA_DIR, seed=SEED)\n",
                        "\n",
                        "# Load real data (Proteins, Drugs, Interactions) from PDB\n",
                        "print(f\"\\nLoading real data from: {DATA_DIR}\")\n",
                        "counts = processor.load_real_data(data_dir=DATA_DIR, max_samples=MAX_DRUGS)\n",
                        "\n",
                        "if counts[\"interactions\"] == 0:\n",
                        "    print(\"\\nWarning: No interactions loaded. Pipeline might fail.\")\n",
                        "\n",
                        "# Get statistics\n",
                        "stats = processor.get_statistics()\n",
                        "print(\"\\nDataset Statistics:\")\n",
                        "for key, value in stats.items():\n",
                        "    if isinstance(value, float):\n",
                        "        print(f\"{key:25s}: {value:.4f}\")\n",
                        "    else:\n",
                        "        print(f\"{key:25s}: {value}\")\n"
                    ]
                    
                    cell['source'] = new_source
                    updated = True
                    break
        
        if updated:
            with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print(f"Successfully updated {NOTEBOOK_PATH}")
        else:
            print("Could not find the data loading cell to update.")
            
    except Exception as e:
        print(f"Error updating notebook: {e}")
        sys.exit(1)

if __name__ == "__main__":
    update_notebook()
