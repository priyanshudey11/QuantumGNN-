import json

nb_path = '/home/priyanshu/QuantumGNN-/Actuall/train_model_quantum.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'MAX_DRUGS =' in source:
            new_source = []
            for line in cell['source']:
                if 'MAX_DRUGS =' in line:
                    new_source.append('MAX_DRUGS = 100          # Limit to 100 proteins for quantum simulation speed\n')
                elif 'BATCH_SIZE =' in line:
                    new_source.append('BATCH_SIZE = 16          # Small batch size for quantum simulation\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print("Updated MAX_DRUGS and BATCH_SIZE")

# Add a markdown cell explaining the limitation
warning_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## \u26a0\ufe0f Important Note on Quantum Simulation Speed\n",
        "\n",
        "Simulating quantum circuits on classical hardware is **computationally expensive**.\n",
        "To ensure this notebook runs in a reasonable time (minutes instead of hours), we have limited the dataset size:\n",
        "- `MAX_DRUGS = 100` (Loads ~100 proteins and their associated drugs)\n",
        "- `BATCH_SIZE = 16`\n",
        "\n",
        "For full-scale training, you would need access to real quantum hardware or a high-performance simulator (like `lightning.gpu`)."
    ]
}

# Insert warning after the first markdown cell
nb['cells'].insert(1, warning_cell)

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook optimized for speed.")
