import json

nb_path = '/home/priyanshu/QuantumGNN-/Actuall/train_model_quantum.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
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
            print("Updated DataLoaders with num_workers=4 and pin_memory=True")

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with optimized DataLoaders.")
