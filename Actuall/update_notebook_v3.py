import json
import os

nb_path = '/home/priyanshu/QuantumGNN-/Actuall/train_model_spark.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

# 1. Update Imports (Cell 2)
import_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'import importlib' in ''.join(cell['source']):
        import_cell = cell
        break

if import_cell:
    new_imports = [
        "from sklearn.model_selection import train_test_split\n",
        "from torch.utils.data import Dataset, DataLoader\n"
    ]
    new_source = []
    for line in import_cell['source']:
        new_source.append(line)
        if "from datetime import datetime" in line:
            new_source.extend(new_imports)
    import_cell['source'] = new_source
    print("Updated Imports")

# 2. Update Split (Cell 8)
split_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'print("Train/Validation Split")' in ''.join(cell['source']):
        split_cell = cell
        break

if split_cell:
    split_cell['source'] = [
        "\n",
        "print(\"Train/Validation Split\")\n",
        "\n",
        "# Check label distribution before split\n",
        "print(\"\\nOverall Label Distribution:\")\n",
        "print(df_pandas['label'].value_counts())\n",
        "\n",
        "# Stratified Split using sklearn\n",
        "train_pd, val_pd = train_test_split(\n",
        "    df_pandas,\n",
        "    test_size=VAL_SPLIT,\n",
        "    random_state=SEED,\n",
        "    stratify=df_pandas['label']\n",
        ")\n",
        "\n",
        "print(\"\\nTrain Label Distribution:\")\n",
        "print(train_pd['label'].value_counts())\n",
        "print(\"\\nValidation Label Distribution:\")\n",
        "print(val_pd['label'].value_counts())\n",
        "\n",
        "# Convert to Spark DataFrame (optional)\n",
        "train_df = spark.createDataFrame(train_pd)\n",
        "val_df = spark.createDataFrame(val_pd)\n",
        "\n",
        "train_df = train_df.repartition(max(1, len(train_pd) // PARTITION_SIZE)).cache()\n",
        "val_df = val_df.repartition(max(1, len(val_pd) // PARTITION_SIZE)).cache()\n",
        "\n",
        "print(f\"\\nTraining samples: {len(train_pd)}\")\n",
        "print(f\"Validation samples: {len(val_pd)}\")\n",
        "\n",
        "# Create PyTorch Datasets and Loaders\n",
        "print(\"\\nCreating PyTorch DataLoaders\")\n",
        "\n",
        "class InteractionDataset(Dataset):\n",
        "    def __init__(self, df):\n",
        "        self.drug_features = np.stack(df['drug_features'].values)\n",
        "        self.patient_features = np.stack(df['patient_features'].values)\n",
        "        self.labels = df['label'].values.astype(np.float32)\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.labels)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        return (\n",
        "            torch.tensor(self.drug_features[idx], dtype=torch.float32),\n",
        "            torch.tensor(self.patient_features[idx], dtype=torch.float32),\n",
        "            torch.tensor(self.labels[idx], dtype=torch.float32),\n",
        "        )\n",
        "\n",
        "train_dataset = InteractionDataset(train_pd)\n",
        "val_dataset = InteractionDataset(val_pd)\n",
        "\n",
        "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)\n",
        "val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)\n"
    ]
    print("Updated Split Cell")

# 3. Update Training Functions (Cell 10)
funcs_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def train_epoch_spark' in ''.join(cell['source']):
        funcs_cell = cell
        break

if funcs_cell:
    funcs_cell['source'] = [
        "def train_epoch(model, optimizer, criterion, loader, device):\n",
        "    \"\"\"Train one epoch using PyTorch DataLoader.\"\"\"\n",
        "    model.train()\n",
        "    total_loss = 0.0\n",
        "    all_preds = []\n",
        "    all_labels = []\n",
        "    \n",
        "    for drug_features, patient_features, labels in loader:\n",
        "        drug_features = drug_features.to(device)\n",
        "        patient_features = patient_features.to(device)\n",
        "        labels = labels.to(device)\n",
        "        \n",
        "        optimizer.zero_grad()\n",
        "        outputs = model(drug_features, patient_features).squeeze(-1)\n",
        "        loss = criterion(outputs, labels)\n",
        "        \n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        \n",
        "        batch_size_actual = len(labels)\n",
        "        total_loss += loss.item() * batch_size_actual\n",
        "        all_preds.extend(outputs.detach().cpu().numpy())\n",
        "        all_labels.extend(labels.cpu().numpy())\n",
        "    \n",
        "    from sklearn.metrics import accuracy_score\n",
        "    avg_loss = total_loss / len(all_labels)\n",
        "    accuracy = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))\n",
        "    \n",
        "    return {'loss': avg_loss, 'accuracy': accuracy}\n",
        "\n",
        "def evaluate(model, criterion, loader, device):\n",
        "    \"\"\"Evaluate model using PyTorch DataLoader.\"\"\"\n",
        "    model.eval()\n",
        "    total_loss = 0.0\n",
        "    all_preds = []\n",
        "    all_labels = []\n",
        "    \n",
        "    with torch.no_grad():\n",
        "        for drug_features, patient_features, labels in loader:\n",
        "            drug_features = drug_features.to(device)\n",
        "            patient_features = patient_features.to(device)\n",
        "            labels = labels.to(device)\n",
        "            \n",
        "            outputs = model(drug_features, patient_features).squeeze(-1)\n",
        "            loss = criterion(outputs, labels)\n",
        "            \n",
        "            batch_size_actual = len(labels)\n",
        "            total_loss += loss.item() * batch_size_actual\n",
        "            all_preds.extend(outputs.cpu().numpy())\n",
        "            all_labels.extend(labels.cpu().numpy())\n",
        "    \n",
        "    from sklearn.metrics import accuracy_score, roc_auc_score\n",
        "    avg_loss = total_loss / len(all_labels)\n",
        "    all_preds = np.array(all_preds)\n",
        "    all_labels = np.array(all_labels)\n",
        "    \n",
        "    accuracy = accuracy_score(all_labels, (all_preds >= 0.5).astype(int))\n",
        "    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else float('nan')\n",
        "    \n",
        "    return {'loss': avg_loss, 'accuracy': accuracy, 'auc': auc}\n",
        "\n",
        "print(\"\\n✓ Training functions defined\")\n"
    ]
    print("Updated Training Functions")

# 4. Update Main Loop (Cell 11)
loop_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'train_epoch_spark' in ''.join(cell['source']):
        loop_cell = cell
        break

if loop_cell:
    loop_cell['source'] = [
        "\n",
        "\n",
        "# Setup training\n",
        "device = torch.device(DEVICE if DEVICE else (\n",
        "    'cuda' if torch.cuda.is_available() else\n",
        "    'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else\n",
        "    'cpu'\n",
        "))\n",
        "\n",
        "model = model.to(device)\n",
        "optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)\n",
        "criterion = torch.nn.BCELoss()\n",
        "\n",
        "print(f\"\\nDevice: {device}\")\n",
        "print(f\"Optimizer: Adam (lr={LEARNING_RATE})\")\n",
        "print(f\"Loss: Binary Cross-Entropy\")\n",
        "print(\"\\nStarting training...\\n\")\n",
        "\n",
        "# Training history\n",
        "history = {\n",
        "    'train_loss': [],\n",
        "    'train_acc': [],\n",
        "    'val_loss': [],\n",
        "    'val_acc': [],\n",
        "    'val_auc': []\n",
        "}\n",
        "\n",
        "# Training loop\n",
        "best_val_loss = float('inf')\n",
        "patience_counter = 0\n",
        "\n",
        "for epoch in range(EPOCHS):\n",
        "    epoch_start = datetime.now()\n",
        "    \n",
        "    # Train\n",
        "    train_metrics = train_epoch(\n",
        "        model, optimizer, criterion, train_loader, device\n",
        "    )\n",
        "    \n",
        "    # Validate\n",
        "    val_metrics = evaluate(\n",
        "        model, criterion, val_loader, device\n",
        "    )\n",
        "    \n",
        "    # Update history\n",
        "    history['train_loss'].append(train_metrics['loss'])\n",
        "    history['train_acc'].append(train_metrics['accuracy'])\n",
        "    history['val_loss'].append(val_metrics['loss'])\n",
        "    history['val_acc'].append(val_metrics['accuracy'])\n",
        "    history['val_auc'].append(val_metrics['auc'])\n",
        "    \n",
        "    epoch_time = (datetime.now() - epoch_start).total_seconds()\n",
        "    \n",
        "    # Print progress\n",
        "    if VERBOSE >= 2:\n",
        "        print(f\"Epoch {epoch+1}/{EPOCHS} - {epoch_time:.2f}s - \"\n",
        "              f\"train_loss: {train_metrics['loss']:.4f} - \"\n",
        "              f\"train_acc: {train_metrics['accuracy']:.4f} - \"\n",
        "              f\"val_loss: {val_metrics['loss']:.4f} - \"\n",
        "              f\"val_acc: {val_metrics['accuracy']:.4f} - \"\n",
        "              f\"val_auc: {val_metrics['auc']:.4f}\")\n",
        "    elif VERBOSE == 1 and (epoch + 1) % 10 == 0:\n",
        "        print(f\"Epoch {epoch+1}/{EPOCHS} - \"\n",
        "              f\"val_loss: {val_metrics['loss']:.4f} - \"\n",
        "              f\"val_acc: {val_metrics['accuracy']:.4f}\")\n",
        "    \n",
        "    # Early stopping\n",
        "    if EARLY_STOPPING is not None:\n",
        "        if val_metrics['loss'] < best_val_loss:\n",
        "            best_val_loss = val_metrics['loss']\n",
        "            patience_counter = 0\n",
        "        else:\n",
        "            patience_counter += 1\n",
        "        \n",
        "        if patience_counter >= EARLY_STOPPING:\n",
        "            print(f\"\\nEarly stopping triggered at epoch {epoch+1}\")\n",
        "            break\n",
        "\n",
        "print(\"Training Complete!\")\n"
    ]
    print("Updated Main Loop")

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
