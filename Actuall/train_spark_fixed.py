
# Drug-Patient QGNN Training with PySpark (Fixed Data Loading)

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Spark Configuration
SPARK_MASTER = "local[*]"        # "local[*]" = all cores, or "spark://host:port"
SPARK_MEMORY = "8g"              # Driver memory
SPARK_EXECUTOR_MEMORY = "4g"     # Executor memory
NUM_EXECUTORS = 6                # Number of executors (for cluster mode)

# Data paths
DATA_DIR = "/media/priyanshu/SD/othercode/data"  # Path to PDB data
SAVE_DIR = "./saved_models"                      # Where to save models

# Data parameters
MAX_DRUGS = None             # Max drugs to load (None = all)
N_PATIENTS = 200000          # Number of synthetic patients
INTERACTION_RATE = 0.05      # Fraction of drug-patient pairs to create

# Model parameters
NUM_QUBITS = 6               # Qubits per side (total = 2 * NUM_QUBITS)
NUM_QLAYERS = 2              # Number of variational layers
HIDDEN_DIM = 64              # Hidden dimension for encoders
USE_QUANTUM = False          # Note: Quantum mode not fully parallelizable with Spark

# Training parameters
EPOCHS = 100                 # Number of training epochs
BATCH_SIZE = 64              # Batch size (larger for distributed)
LEARNING_RATE = 0.001        # Learning rate
VAL_SPLIT = 0.2              # Validation split (0.2 = 20%)
EARLY_STOPPING = None        # Early stopping patience (None = disabled)

# Spark-specific parameters
PARTITION_SIZE = 100         # Samples per partition
SHUFFLE_PARTITIONS = 200     # Number of shuffle partitions

# Other parameters
SEED = 42                    # Random seed
DEVICE = None                # Device (None = auto-detect, 'cuda', 'mps', 'cpu')
VERBOSE = 2                  # Verbosity (0=silent, 1=progress, 2=detailed)

# ============================================================================

# PySpark imports
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, udf, pandas_udf, PandasUDFType, rand as spark_rand
    from pyspark.sql.types import *
    from pyspark import SparkContext, SparkConf
    PYSPARK_AVAILABLE = True
except ImportError:
    print(" PySpark not installed. Installing...")
    print("Run: pip install pyspark")
    PYSPARK_AVAILABLE = False

# Import drug-patient QGNN
from drug_patient_qgnn import (
    DrugPatientDataProcessor,
    QuantumDrugPatientGNN,
    DrugPatientTrainer,
    set_seed,
    print_model_summary,
    print_device_info,
    plot_training_history,
    calculate_metrics,
    print_metrics,
    save_training_history
)

# Set random seed
set_seed(SEED)

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)

# Create experiment name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
mode = "quantum" if USE_QUANTUM else "classical"
EXPERIMENT_NAME = f"spark_{mode}_q{NUM_QUBITS}_l{NUM_QLAYERS}_{timestamp}"

print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Mode: {'QUANTUM' if USE_QUANTUM else 'CLASSICAL'} + SPARK")
print(f"Spark Master: {SPARK_MASTER}")

# Configure Spark
if not PYSPARK_AVAILABLE:
    raise ImportError("PySpark is required. Install with: pip install pyspark")

conf = SparkConf() \
    .setAppName("DrugPatientQGNN") \
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
print_device_info()

# ============================================================================
# Step 1: Load and Process Data
# ============================================================================
print("\nLoading Data with Spark")

# Initialize processor
processor = DrugPatientDataProcessor(data_dir=DATA_DIR, seed=SEED)

# Load real data (Proteins, Drugs, Interactions) from PDB
print(f"\nLoading real data from: {DATA_DIR}")
counts = processor.load_real_data(data_dir=DATA_DIR, max_samples=MAX_DRUGS)

if counts["interactions"] == 0:
    print("\nWarning: No interactions loaded. Pipeline might fail.")

# Get statistics
stats = processor.get_statistics()
print("\nDataset Statistics:")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"{key:25s}: {value:.4f}")
    else:
        print(f"{key:25s}: {value}")

# ============================================================================
# Step 2: Create DataFrames and Split
# ============================================================================
print("\nCreating DataFrames and Splitting")

graph = processor.graph
drug_features = graph.get_drug_features_matrix()
patient_features = graph.get_patient_features_matrix()
edge_index, edge_features = graph.get_edge_index()
labels = graph.get_edge_labels()

# Create interaction dataset
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

# Create pandas DataFrame
df_pandas = pd.DataFrame(interaction_data)

# Check label distribution before split
print("\nOverall Label Distribution:")
print(df_pandas['label'].value_counts())

# Stratified Split using sklearn
train_pd, val_pd = train_test_split(
    df_pandas,
    test_size=VAL_SPLIT,
    random_state=SEED,
    stratify=df_pandas['label']
)

print("\nTrain Label Distribution:")
print(train_pd['label'].value_counts())
print("\nValidation Label Distribution:")
print(val_pd['label'].value_counts())

# Convert to Spark DataFrame (optional, for future use or hybrid approach)
train_df_spark = spark.createDataFrame(train_pd)
val_df_spark = spark.createDataFrame(val_pd)

train_df_spark = train_df_spark.repartition(max(1, len(train_pd) // PARTITION_SIZE)).cache()
val_df_spark = val_df_spark.repartition(max(1, len(val_pd) // PARTITION_SIZE)).cache()

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
    use_quantum=USE_QUANTUM
)

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
        outputs = model(drug_features, patient_features).squeeze(-1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        batch_size_actual = len(labels)
        total_loss += loss.item() * batch_size_actual
        all_preds.extend(outputs.detach().cpu().numpy())
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
            
            outputs = model(drug_features, patient_features).squeeze(-1)
            loss = criterion(outputs, labels)
            
            batch_size_actual = len(labels)
            total_loss += loss.item() * batch_size_actual
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import accuracy_score, roc_auc_score
    avg_loss = total_loss / len(all_labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, (all_preds >= 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else float('nan')
    
    return {'loss': avg_loss, 'accuracy': accuracy, 'auc': auc}

# Setup training
device = torch.device(DEVICE if DEVICE else (
    'cuda' if torch.cuda.is_available() else
    'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
    'cpu'
))

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()

print(f"\nDevice: {device}")
print("Starting training...\n")

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [], 'val_auc': []
}

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    epoch_start = datetime.now()
    
    train_metrics = train_epoch(
        model, optimizer, criterion, train_loader, device
    )
    
    val_metrics = evaluate(
        model, criterion, val_loader, device
    )
    
    history['train_loss'].append(train_metrics['loss'])
    history['train_acc'].append(train_metrics['accuracy'])
    history['val_loss'].append(val_metrics['loss'])
    history['val_acc'].append(val_metrics['accuracy'])
    history['val_auc'].append(val_metrics['auc'])
    
    epoch_time = (datetime.now() - epoch_start).total_seconds()
    
    if VERBOSE >= 2:
        print(f"Epoch {epoch+1}/{EPOCHS} - {epoch_time:.2f}s - "
              f"train_loss: {train_metrics['loss']:.4f} - "
              f"train_acc: {train_metrics['accuracy']:.4f} - "
              f"val_loss: {val_metrics['loss']:.4f} - "
              f"val_acc: {val_metrics['accuracy']:.4f} - "
              f"val_auc: {val_metrics['auc']:.4f}")
    
    if EARLY_STOPPING is not None:
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOPPING:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

print("Training Complete!")

# Save Results
print("\nSaving Results")
model_path = os.path.join(SAVE_DIR, f"{EXPERIMENT_NAME}_model.pt")
torch.save(model.state_dict(), model_path)
print(f"✓ Model saved: {model_path}")

graph_path = os.path.join(SAVE_DIR, f"{EXPERIMENT_NAME}_graph.pkl")
processor.save_graph(graph_path)
print(f"✓ Graph saved: {graph_path}")

history_path = os.path.join(SAVE_DIR, f"{EXPERIMENT_NAME}_history.json")
save_training_history(history, history_path)
print(f"✓ History saved: {history_path}")
