import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# PySpark imports
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, rand as spark_rand
    from pyspark import SparkContext, SparkConf
    PYSPARK_AVAILABLE = True
except ImportError:
    print("PySpark not installed. Running in local mode only.")
    PYSPARK_AVAILABLE = False

# Import drug-patient QGNN
from drug_patient_qgnn import (
    DrugPatientDataProcessor,
    QuantumDrugPatientGNN,
    set_seed,
    print_model_summary,
    print_device_info
)

# Check for PennyLane and GPU
try:
    import pennylane as qml
    if 'lightning.gpu' in qml.plugin_devices:
        DEVICE_NAME = 'lightning.gpu'
        print("✓ lightning.gpu available. Using GPU for quantum simulation.")
    elif 'lightning.qubit' in qml.plugin_devices:
        DEVICE_NAME = 'lightning.qubit'
        print("⚠ lightning.gpu not found. Using lightning.qubit (Fast CPU).")
    else:
        DEVICE_NAME = 'default.qubit'
        print("⚠ Using default.qubit (Slow CPU).")
except ImportError:
    DEVICE_NAME = 'default.qubit'

# ============================================================================
# Configuration
# ============================================================================
# Spark Configuration
SPARK_MASTER = "local[*]"
SPARK_MEMORY = "8g"
SPARK_EXECUTOR_MEMORY = "4g"
SHUFFLE_PARTITIONS = 200

# Data paths
DATA_DIR = "/media/priyanshu/SD/othercode/data"
SAVE_DIR = "./saved_models"

# Data parameters
MAX_DRUGS = 2000             # Increased for GPU
INTERACTION_RATE = 0.05

# Model parameters
NUM_QUBITS = 6               # Increased for GPU
NUM_QLAYERS = 2
HIDDEN_DIM = 64
USE_QUANTUM = True

# Training parameters
EPOCHS = 20
BATCH_SIZE = 128             # Increased for GPU
LEARNING_RATE = 0.0001       # Lower LR for quantum stability
VAL_SPLIT = 0.2
EARLY_STOPPING = None

# Other parameters
SEED = 42
PARTITION_SIZE = 100

# Set random seed
set_seed(SEED)

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================================
# Step 0: Initialize Spark
# ============================================================================
if PYSPARK_AVAILABLE:
    conf = SparkConf() \
        .setAppName("DrugPatientQGNN_GPU") \
        .setMaster(SPARK_MASTER) \
        .set("spark.driver.memory", SPARK_MEMORY) \
        .set("spark.executor.memory", SPARK_EXECUTOR_MEMORY) \
        .set("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS)) \
        .set("spark.default.parallelism", str(SHUFFLE_PARTITIONS))

    spark = SparkSession.builder \
        .config(conf=conf) \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("WARN")
    print("Spark Session Initialized")
else:
    print("Skipping Spark initialization")

print_device_info()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Step 1: Load and Process Data
# ============================================================================
print("\nLoading Data")
processor = DrugPatientDataProcessor(data_dir=DATA_DIR, seed=SEED)
counts = processor.load_real_data(data_dir=DATA_DIR, max_samples=MAX_DRUGS)

if counts["interactions"] == 0:
    print("\nWarning: No interactions loaded. Pipeline might fail.")

# ============================================================================
# Step 2: Create DataFrames and Split
# ============================================================================
print("\nCreating DataFrames and Splitting")

graph = processor.graph
drug_features = graph.get_drug_features_matrix()
patient_features = graph.get_patient_features_matrix()
edge_index, edge_features = graph.get_edge_index()
labels = graph.get_edge_labels()

interaction_data = []
for idx in range(edge_index.shape[1]):
    drug_idx = int(edge_index[0, idx])
    patient_idx = int(edge_index[1, idx])
    
    interaction_data.append({
        'interaction_id': idx,
        'drug_idx': drug_idx,
        'patient_idx': patient_idx,
        'drug_features': drug_features[drug_idx].tolist(),
        'patient_features': patient_features[patient_idx].tolist(),
        'label': float(labels[idx])
    })

df_pandas = pd.DataFrame(interaction_data)

# Stratified Split
train_pd, val_pd = train_test_split(
    df_pandas,
    test_size=VAL_SPLIT,
    random_state=SEED,
    stratify=df_pandas['label']
)

print(f"\nTraining samples: {len(train_pd)}")
print(f"Validation samples: {len(val_pd)}")

# ============================================================================
# Step 3: Create PyTorch Datasets and Loaders
# ============================================================================
print("\nCreating PyTorch DataLoaders")

class InteractionDataset(Dataset):
    def __init__(self, df):
        self.drug_features = np.stack(df['drug_features'].values)
        self.patient_features = np.stack(df['patient_features'].values)
        self.labels = df['label'].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.drug_features[idx], dtype=torch.float32),
            torch.tensor(self.patient_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

train_dataset = InteractionDataset(train_pd)
val_dataset = InteractionDataset(val_pd)

# Optimized DataLoaders for GPU
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ============================================================================
# Step 4: Create Model
# ============================================================================
print("\nCreating Model")

drug_dim = len(drug_features[0])
patient_dim = len(patient_features[0])

model = QuantumDrugPatientGNN(
    drug_dim=drug_dim,
    patient_dim=patient_dim,
    num_qubits=NUM_QUBITS,
    num_qlayers=NUM_QLAYERS,
    hidden_dim=HIDDEN_DIM,
    use_quantum=USE_QUANTUM,
    device_name=DEVICE_NAME  # Pass GPU device
)

model = model.to(device)
print_model_summary(model, drug_dim, patient_dim)

# ============================================================================
# Step 5: Training Loop
# ============================================================================

def train_epoch(model, optimizer, criterion, loader, device):
    """Train one epoch using PyTorch DataLoader."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for drug_features, patient_features, labels in loader:
        drug_features = drug_features.to(device)
        patient_features = patient_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        # Model returns logits now
        outputs = model(drug_features, patient_features).squeeze(-1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        batch_size_actual = len(labels)
        total_loss += loss.item() * batch_size_actual
        
        # Apply sigmoid for metrics
        probs = torch.sigmoid(outputs)
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import accuracy_score
    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
    
    return {'loss': avg_loss, 'accuracy': accuracy}

def evaluate(model, criterion, loader, device):
    """Evaluate model using PyTorch DataLoader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for drug_features, patient_features, labels in loader:
            drug_features = drug_features.to(device)
            patient_features = patient_features.to(device)
            labels = labels.to(device)
            
            # Model returns logits
            outputs = model(drug_features, patient_features).squeeze(-1)
            loss = criterion(outputs, labels)
            
            batch_size_actual = len(labels)
            total_loss += loss.item() * batch_size_actual
            
            # Apply sigmoid for metrics
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import accuracy_score, roc_auc_score
    avg_loss = total_loss / len(all_labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, (all_preds >= 0.5).astype(int))
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = float('nan')
    
    return {'loss': avg_loss, 'accuracy': accuracy, 'auc': auc}

# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()  # Use Logits Loss

print(f"\nStarting training on {device}...")
print(f"Quantum Device: {DEVICE_NAME}")

for epoch in range(EPOCHS):
    start_time = datetime.now()
    
    train_metrics = train_epoch(model, optimizer, criterion, train_loader, device)
    val_metrics = evaluate(model, criterion, val_loader, device)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    print(f"Epoch {epoch+1}/{EPOCHS} - {duration:.2f}s - "
          f"train_loss: {train_metrics['loss']:.4f} - train_acc: {train_metrics['accuracy']:.4f} - "
          f"val_loss: {val_metrics['loss']:.4f} - val_acc: {val_metrics['accuracy']:.4f} - "
          f"val_auc: {val_metrics['auc']:.4f}")

print("\nTraining complete.")
