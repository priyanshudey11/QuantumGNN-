import json

nb_path = '/home/priyanshu/QuantumGNN-/Actuall/train_model_quantum.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Update Loss Function
        if 'criterion = torch.nn.BCELoss()' in source:
            cell['source'] = [line.replace('criterion = torch.nn.BCELoss()', 'criterion = torch.nn.BCEWithLogitsLoss()') for line in cell['source']]
            print("Updated Loss: BCEWithLogitsLoss")
            
        # Update Learning Rate
        if 'LEARNING_RATE =' in source:
            new_source = []
            for line in cell['source']:
                if 'LEARNING_RATE =' in line:
                    new_source.append('LEARNING_RATE = 0.0001       # Lower LR for quantum stability\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print("Updated LR: 0.0001")

        # Update Training Loop (Sigmoid for accuracy calculation)
        if 'outputs = model(drug_features, patient_features).squeeze(-1)' in source:
             # We need to apply sigmoid to outputs for accuracy calculation because model now returns logits
             # But wait, accuracy calculation is done later:
             # accuracy = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
             # all_preds comes from outputs.detach().cpu().numpy()
             # So we need to sigmoid the outputs before adding to all_preds
             
             new_source = []
             for line in cell['source']:
                 new_source.append(line)
                 if 'outputs = model(drug_features, patient_features).squeeze(-1)' in line:
                     # This is fine for loss calculation (BCEWithLogitsLoss takes logits)
                     pass
                 if 'all_preds.extend(outputs.detach().cpu().numpy())' in line:
                     # Replace this line to apply sigmoid
                     new_source.pop() # Remove the last added line
                     new_source.append('        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())\n')
                 if 'all_preds.extend(outputs.cpu().numpy())' in line: # For evaluate function
                     new_source.pop()
                     new_source.append('            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())\n')
             
             cell['source'] = new_source
             print("Updated Training Loop: Applied sigmoid for metrics")

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with architectural refinements.")
