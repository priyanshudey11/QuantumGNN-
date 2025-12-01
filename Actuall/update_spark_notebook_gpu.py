import json

nb_path = '/home/priyanshu/QuantumGNN-/Actuall/train_model_spark.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # 1. Update Parameters
        if 'USE_QUANTUM =' in source:
            new_source = []
            for line in cell['source']:
                if 'USE_QUANTUM =' in line:
                    new_source.append('USE_QUANTUM = True           # Enabled for GPU training\n')
                elif 'MAX_DRUGS =' in line:
                    new_source.append('MAX_DRUGS = 2000             # Increased for GPU\n')
                elif 'BATCH_SIZE =' in line:
                    new_source.append('BATCH_SIZE = 128             # Increased for GPU\n')
                elif 'NUM_QUBITS =' in line:
                    new_source.append('NUM_QUBITS = 6               # Increased for GPU\n')
                elif 'LEARNING_RATE =' in line:
                    new_source.append('LEARNING_RATE = 0.0001       # Lower LR for quantum stability\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print("Updated Parameters: QUANTUM=True, MAX_DRUGS=2000, BATCH=128, QUBITS=6, LR=0.0001")

        # 2. Add GPU Device Detection
        if 'DEVICE = None' in source:
             # Insert GPU detection code after imports or configuration
             pass # We'll handle this by modifying the model instantiation block

        # 3. Update Model Instantiation to use GPU
        if 'model = QuantumDrugPatientGNN(' in source:
            # Add device detection before model creation
            gpu_check = [
                "\n",
                "# Check for GPU Device\n",
                "try:\n",
                "    import pennylane as qml\n",
                "    if 'lightning.gpu' in qml.plugin_devices:\n",
                "        DEVICE_NAME = 'lightning.gpu'\n",
                "        print(\"✓ lightning.gpu available. Using GPU for quantum simulation.\")\n",
                "    elif 'lightning.qubit' in qml.plugin_devices:\n",
                "        DEVICE_NAME = 'lightning.qubit'\n",
                "        print(\"⚠ lightning.gpu not found. Using lightning.qubit (Fast CPU).\")\n",
                "    else:\n",
                "        DEVICE_NAME = 'default.qubit'\n",
                "        print(\"⚠ Using default.qubit (Slow CPU).\")\n",
                "except ImportError:\n",
                "    DEVICE_NAME = 'default.qubit'\n",
                "\n"
            ]
            
            # Prepend GPU check to the cell
            cell['source'] = gpu_check + cell['source']
            
            # Add device_name to model init
            new_source = []
            for line in cell['source']:
                if 'use_quantum=USE_QUANTUM' in line:
                    new_source.append(line.rstrip() + ',\n')
                    new_source.append('    device_name=DEVICE_NAME  # Pass GPU device\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print("Updated Model Instantiation: Added DEVICE_NAME")

        # 4. Update DataLoaders
        if 'DataLoader(' in source:
            new_source = []
            for line in cell['source']:
                if 'train_loader = DataLoader' in line:
                    new_source.append('train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)\n')
                elif 'val_loader = DataLoader' in line:
                    new_source.append('val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print("Updated DataLoaders: Added num_workers=4, pin_memory=True")

        # 5. Update Loss Function
        if 'criterion = torch.nn.BCELoss()' in source:
            cell['source'] = [line.replace('criterion = torch.nn.BCELoss()', 'criterion = torch.nn.BCEWithLogitsLoss()') for line in cell['source']]
            print("Updated Loss: BCEWithLogitsLoss")

        # 6. Update Training Loop (Logits handling)
        if 'outputs = model(drug_features, patient_features).squeeze(-1)' in source:
             new_source = []
             for line in cell['source']:
                 new_source.append(line)
                 if 'outputs = model(drug_features, patient_features).squeeze(-1)' in line:
                     pass # Logits are fine for loss
                 if 'all_preds.extend(outputs.detach().cpu().numpy())' in line:
                     new_source.pop()
                     new_source.append('        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())\n')
                 if 'all_preds.extend(outputs.cpu().numpy())' in line:
                     new_source.pop()
                     new_source.append('            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())\n')
             
             cell['source'] = new_source
             print("Updated Training Loop: Applied sigmoid for metrics")

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Spark Notebook updated for GPU acceleration.")
