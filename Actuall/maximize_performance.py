import json

nb_path = '/home/priyanshu/QuantumGNN-/Actuall/train_model_quantum.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Update Parameters
        if 'MAX_DRUGS =' in source:
            new_source = []
            for line in cell['source']:
                if 'MAX_DRUGS =' in line:
                    new_source.append('MAX_DRUGS = 1000         # Increased for better generalization\n')
                elif 'BATCH_SIZE =' in line:
                    new_source.append('BATCH_SIZE = 128         # Increased to utilize GPU better\n')
                elif 'NUM_QUBITS =' in line:
                     new_source.append('NUM_QUBITS = 4           # Reduced qubits for speed (Total 8)\n')
                elif 'NUM_QLAYERS =' in line:
                     new_source.append('NUM_QLAYERS = 1          # Reduced layers for speed\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print("Updated Parameters: MAX_DRUGS=1000, BATCH_SIZE=128, QUBITS=4, LAYERS=1")

        # Update Quantum Device to lightning.qubit
        if 'qml.device' in source:
            cell['source'] = [line.replace('default.qubit', 'lightning.qubit') for line in cell['source']]
            print("Updated Device: Used lightning.qubit")

# Add explanation about GPU usage
gpu_note = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## \u26a1 Performance Optimization\n",
        "\n",
        "To maximize performance without `lightning.gpu` (which requires cuQuantum):\n",
        "1.  **Simulator**: Switched to `lightning.qubit` (Fast C++ CPU backend).\n",
        "2.  **Batch Size**: Increased to **128** to push more data to the GPU.\n",
        "3.  **Circuit**: Simplified to 4 qubits/1 layer to prevent CPU bottlenecks.\n",
        "\n",
        "**Note**: You likely won't see 90% GPU usage because the quantum simulation (CPU) is still the bottleneck. This is normal for quantum-classical hybrids without specific GPU-quantum support."
    ]
}
nb['cells'].insert(2, gpu_note)

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook configured for maximum performance.")
