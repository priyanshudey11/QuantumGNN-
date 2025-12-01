import json

nb_path = '/home/priyanshu/QuantumGNN-/Actuall/train_model_quantum.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Update Parameters for GPU
        if 'MAX_DRUGS =' in source:
            new_source = []
            for line in cell['source']:
                if 'MAX_DRUGS =' in line:
                    new_source.append('MAX_DRUGS = 2000         # Increased for GPU\n')
                elif 'NUM_QUBITS =' in line:
                     new_source.append('NUM_QUBITS = 6           # Increased back to 6 for GPU\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print("Updated Parameters: MAX_DRUGS=2000, QUBITS=6")

        # Update Quantum Device to lightning.gpu
        if 'qml.device' in source:
            cell['source'] = [line.replace('lightning.qubit', 'lightning.gpu') for line in cell['source']]
            # Also handle if it was default.qubit
            cell['source'] = [line.replace('default.qubit', 'lightning.gpu') for line in cell['source']]
            print("Updated Device: Used lightning.gpu")

# Update the GPU note
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and 'Performance Optimization' in ''.join(cell['source']):
        cell['source'] = [
            "## \u26a1 GPU Acceleration Enabled\n",
            "\n",
            "**`lightning.gpu` is active!**\n",
            "The quantum simulation is now running on your NVIDIA GPU.\n",
            "- **Simulator**: `lightning.gpu` (cuQuantum)\n",
            "- **Workload**: Increased to 6 qubits and 2000 samples to utilize the GPU.\n"
        ]

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook configured for GPU acceleration.")
