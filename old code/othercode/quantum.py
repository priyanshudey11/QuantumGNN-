# %% [markdown]
# # Quantum Drug-Protein Interaction Prediction
#
# This notebook trains a **Quantum Graph Neural Network (QGNN)** to predict drug-protein interactions.
# It uses **PennyLane** for the quantum circuit layer.
#


# %% [markdown]
# ## ⚠️ Important Note on Quantum Simulation Speed
#
# Simulating quantum circuits on classical hardware is **computationally expensive**.
# To ensure this notebook runs in a reasonable time (minutes instead of hours), we have limited the dataset size:
# - `MAX_DRUGS = 100` (Loads ~100 proteins and their associated drugs)
# - `BATCH_SIZE = 16`
#
# For full-scale training, you would need access to real quantum hardware or a high-performance simulator (like `lightning.gpu`).


# %%
# Spark Configuration


# Data paths
DATA_DIR = "/media/priyanshu/SD/othercode/data"  # Path to PDB data
SAVE_DIR = "./saved_models"                      # Where to save models


# Data parameters
MAX_DRUGS = 1000    # Limit to 100 proteins for quantum simulation speed
INTERACTION_RATE = 0.05      # Fraction of drug-patient pairs to create


# Model parameters
NUM_QUBITS = 6               # Qubits per side (total = 2 * NUM_QUBITS)
NUM_QLAYERS = 2              # Number of variational layers
HIDDEN_DIM = 64              # Hidden dimension for encoders
USE_QUANTUM = True  # Enable Quantum Mode


# Training parameters
EPOCHS = 20 
BATCH_SIZE = 16          # Small batch size for quantum simulation
LEARNING_RATE = 0.001        # Learning rate
VAL_SPLIT = 0.2              # Validation split (0.2 = 20%)
EARLY_STOPPING = None        # Early stopping patience (None = disabled)


# Spark-specific parameters


# Other parameters
SEED = 42                    # Random seed
DEVICE = None                # Device (None = auto-detect, 'cuda', 'mps', 'cpu')
VERBOSE = 2                  # Verbosity (0=silent, 1=progress, 2=detailed)


# ============================================================================


# %% [markdown]
# ## Setup Imports
#


# %%
import importlib



%load_ext autoreload
%autoreload 2
# PySpark imports
try:
   from pyspark.sql import SparkSession
   from pyspark.sql.functions import col, udf, pandas_udf, PandasUDFType
   from pyspark.sql.types import *
   from pyspark import SparkContext, SparkConf
   PYSPARK_AVAILABLE = True
except ImportError:
   print(" PySpark not installed. Installing...")
   print("Run: pip install pyspark")
   PYSPARK_AVAILABLE = False
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
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
   print_metrics
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
print(f"Qubits: {NUM_QUBITS} per side (total: {2*NUM_QUBITS})")
print(f"Layers: {NUM_QLAYERS}")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")


import drug_patient_qgnn
print(f'Using drug_patient_qgnn from: {drug_patient_qgnn.__file__}')




# %%


import subprocess
import glob


def find_java_home():
   """Try to find JAVA_HOME automatically, prioritizing Java 17."""
   # Common Java installation paths (PRIORITIZE JAVA 17)
   java_paths = [
       '/usr/lib/jvm/java-17-openjdk-amd64',  # Java 17 first
       '/usr/lib/jvm/java-11-openjdk-amd64',
       '/usr/lib/jvm/java-8-openjdk-amd64',
       '/usr/lib/jvm/default-java',
   ]
  
   # Check each path
   for path in java_paths:
       if os.path.exists(path):
           return path
  
   # Try to find Java 17 using glob pattern first
   jvm_dirs_17 = glob.glob('/usr/lib/jvm/java-17*')
   if jvm_dirs_17:
       return sorted(jvm_dirs_17)[-1]
  
   # Try to find using update-alternatives
   try:
       result = subprocess.run(['readlink', '-f', '/usr/bin/java'],
                             capture_output=True, text=True, timeout=5)
       if result.returncode == 0:
           java_bin = result.stdout.strip()
           # Extract JAVA_HOME (remove /bin/java)
           java_home = java_bin.replace('/bin/java', '')
           if os.path.exists(java_home):
               return java_home
   except:
       pass
  
   # Try glob pattern for any Java version
   jvm_dirs = glob.glob('/usr/lib/jvm/java-*-openjdk-*')
   if jvm_dirs:
       return sorted(jvm_dirs)[-1]  # Return newest version
  
   return None


# Check if JAVA_HOME is already set
if 'JAVA_HOME' not in os.environ:
   java_home = find_java_home()
  
   if java_home:
       os.environ['JAVA_HOME'] = java_home
       print(f" JAVA_HOME automatically set to: {java_home}")
   else:
       print("JAVA NOT FOUND - INSTALLATION REQUIRED")
       print("\nPySpark requires Java (JDK 17 recommended). Please install it:")
       print("\nOpen a terminal and run:")
       print("  sudo apt update")
       print("  sudo apt install -y openjdk-17-jdk")
       print("\nAfter installation, restart this notebook.")
      
       raise RuntimeError(
           "Java is not installed. Please install Java and restart the notebook."
       )
else:
   print(f" JAVA_HOME already set: {os.environ['JAVA_HOME']}")


# Verify Java is working
try:
   result = subprocess.run(['java', '-version'],
                         capture_output=True, text=True, timeout=5)
   version_output = result.stderr.split('\n')[0]  # Java version goes to stderr
   print(f" Java version: {version_output}")
  
   # Check if we're using the right Java version
   if 'JAVA_HOME' in os.environ:
       java_executable = os.path.join(os.environ['JAVA_HOME'], 'bin', 'java')
       if os.path.exists(java_executable):
           result_home = subprocess.run([java_executable, '-version'],
                                      capture_output=True, text=True, timeout=5)
           version_home = result_home.stderr.split('\n')[0]
           print(f" JAVA_HOME Java version: {version_home}")
except Exception as e:
   print(f"⚠️  Warning: Could not verify Java installation: {e}")


 # %%
print_device_info()


import pennylane as qml
print(f"PennyLane Version: {qml.__version__}")




# %% [markdown]
# ## Step 1: Load and Process Data with Spark


# %%
print("Loading Data")


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




# %%


print("Creating DataFrames")




# Get graph data
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


# Partition data


# Cache for faster access




# Show sample




# %%


print("Train/Validation Split")


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


# Convert to Spark DataFrame (optional)




print(f"\nTraining samples: {len(train_pd)}")
print(f"Validation samples: {len(val_pd)}")


# Create PyTorch Datasets and Loaders
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




# %%


print("Creating Model")




# Get feature dimensions
drug_dim = len(drug_features[0])
patient_dim = len(patient_features[0])


# Create model
model = QuantumDrugPatientGNN(
   drug_dim=drug_dim,
   patient_dim=patient_dim,
   num_qubits=NUM_QUBITS,
   num_qlayers=NUM_QLAYERS,
   hidden_dim=HIDDEN_DIM,
   use_quantum=USE_QUANTUM
)


# Print model summary
print_model_summary(model, drug_dim, patient_dim)


if USE_QUANTUM:
   print("\n  Note: Quantum mode with Spark may not provide speedup")
   print("   Quantum circuits are not easily parallelizable")
   print("   Consider USE_QUANTUM=False for distributed training")
else:
   print("\n Classical mode")


# %%
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


print("\n Training functions defined")




# %%




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
print(f"Optimizer: Adam (lr={LEARNING_RATE})")
print(f"Loss: Binary Cross-Entropy")
print("\nStarting training...\n")


# Training history
history = {
   'train_loss': [],
   'train_acc': [],
   'val_loss': [],
   'val_acc': [],
   'val_auc': []
}


# Training loop
best_val_loss = float('inf')
patience_counter = 0


for epoch in range(EPOCHS):
   epoch_start = datetime.now()
  
   # Train
   train_metrics = train_epoch(
       model, optimizer, criterion, train_loader, device
   )
  
   # Validate
   val_metrics = evaluate(
       model, criterion, val_loader, device
   )
  
   # Update history
   history['train_loss'].append(train_metrics['loss'])
   history['train_acc'].append(train_metrics['accuracy'])
   history['val_loss'].append(val_metrics['loss'])
   history['val_acc'].append(val_metrics['accuracy'])
   history['val_auc'].append(val_metrics['auc'])
  
   epoch_time = (datetime.now() - epoch_start).total_seconds()
  
   # Print progress
   if VERBOSE >= 2:
       print(f"Epoch {epoch+1}/{EPOCHS} - {epoch_time:.2f}s - "
             f"train_loss: {train_metrics['loss']:.4f} - "
             f"train_acc: {train_metrics['accuracy']:.4f} - "
             f"val_loss: {val_metrics['loss']:.4f} - "
             f"val_acc: {val_metrics['accuracy']:.4f} - "
             f"val_auc: {val_metrics['auc']:.4f}")
   elif VERBOSE == 1 and (epoch + 1) % 10 == 0:
       print(f"Epoch {epoch+1}/{EPOCHS} - "
             f"val_loss: {val_metrics['loss']:.4f} - "
             f"val_acc: {val_metrics['accuracy']:.4f}")
  
   # Early stopping
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




# %%
# Plot training history
plot_training_history(history)


# Print final metrics
print("\nFinal Training Metrics:")


print(f"Train Loss:     {history['train_loss'][-1]:.4f}")
print(f"Train Accuracy: {history['train_acc'][-1]:.4f}")
print(f"Val Loss:       {history['val_loss'][-1]:.4f}")
print(f"Val Accuracy:   {history['val_acc'][-1]:.4f}")
print(f"Val AUC-ROC:    {history['val_auc'][-1]:.4f}")




# %%


print("Saving Results")




# Save model state
model_path = os.path.join(SAVE_DIR, f"{EXPERIMENT_NAME}_model.pt")
torch.save(model.state_dict(), model_path)
print(f" Model saved: {model_path}")


# Save graph
graph_path = os.path.join(SAVE_DIR, f"{EXPERIMENT_NAME}_graph.pkl")
processor.save_graph(graph_path)
print(f" Graph saved: {graph_path}")


# Save training history
from drug_patient_qgnn import save_training_history
history_path = os.path.join(SAVE_DIR, f"{EXPERIMENT_NAME}_history.json")
save_training_history(history, history_path)
print(f" History saved: {history_path}")


# Save configuration
import json
config = {
   'experiment_name': EXPERIMENT_NAME,
   'timestamp': timestamp,
   'mode': mode,
   'num_qubits': NUM_QUBITS,
   'num_qlayers': NUM_QLAYERS,
   'epochs': EPOCHS,
   'batch_size': BATCH_SIZE,
   'learning_rate': LEARNING_RATE,
   'val_split': VAL_SPLIT,
   'seed': SEED,
   'max_drugs': MAX_DRUGS,
   'interaction_rate': INTERACTION_RATE
}


config_path = os.path.join(SAVE_DIR, f"{EXPERIMENT_NAME}_config.json")
with open(config_path, 'w') as f:
   json.dump(config, f, indent=2)
print(f" Config saved: {config_path}")