import json

nb_path = '/home/priyanshu/QuantumGNN-/Actuall/train_model_quantum.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

# 1. Update Title (Cell 0)
if nb['cells'][0]['cell_type'] == 'markdown':
    nb['cells'][0]['source'] = [
        "# Quantum Drug-Protein Interaction Prediction\n",
        "\n",
        "This notebook trains a **Quantum Graph Neural Network (QGNN)** to predict drug-protein interactions.\n",
        "It uses **PennyLane** for the quantum circuit layer.\n"
    ]

# 2. Update Configuration (Find cell with USE_QUANTUM)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'USE_QUANTUM' in source:
            new_source = []
            for line in cell['source']:
                if 'USE_QUANTUM =' in line:
                    new_source.append('USE_QUANTUM = True  # Enable Quantum Mode\n')
                elif 'BATCH_SIZE =' in line:
                    new_source.append('BATCH_SIZE = 32     # Reduced batch size for quantum simulation\n')
                elif 'EPOCHS =' in line:
                    new_source.append('EPOCHS = 10         # Fewer epochs for demonstration\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print("Updated Configuration: Enabled Quantum Mode")

# 3. Add Quantum Device Info (Optional)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'print_device_info()' in ''.join(cell['source']):
        cell['source'].append('\nimport pennylane as qml\nprint(f"PennyLane Version: {qml.__version__}")\n')

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
