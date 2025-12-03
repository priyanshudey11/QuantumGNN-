"""
Training script for Ligand-Pocket QGNN.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from ligand_pocket_qgnn.data import LigandPocketDataProcessor, LigandPocketDataset, collate_fn
from ligand_pocket_qgnn.model import LigandPocketQGNN

# Configuration
DATA_DIR = "/media/priyanshu/SD/othercode/data"
SAVE_DIR = "./ligand_pocket_results"
MAX_SAMPLES = 500
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Load Data
    print("Loading Data...")
    processor = LigandPocketDataProcessor(DATA_DIR)
    processor.load_data(max_samples=MAX_SAMPLES)
    
    interactions = processor.get_dataset()
    
    if len(interactions) == 0:
        print("No interactions found! Check data directory.")
        return
        
    # Split
    train_ints, val_ints = train_test_split(interactions, test_size=0.2, random_state=42)
    
    train_dataset = LigandPocketDataset(processor, train_ints)
    val_dataset = LigandPocketDataset(processor, val_ints)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 2. Initialize Model
    sample_ligand = processor.ligands[interactions[0].ligand_id]
    sample_pocket = processor.pockets[interactions[0].pocket_id]
    
    ligand_dim = sample_ligand.atom_features.shape[1]
    pocket_dim = sample_pocket.to_vector().shape[0]
    
    print(f"Ligand Feature Dim: {ligand_dim}")
    print(f"Pocket Feature Dim: {pocket_dim}")
    
    model = LigandPocketQGNN(
        ligand_in_dim=ligand_dim,
        pocket_in_dim=pocket_dim,
        n_qubits=6,
        n_qlayers=2,
        use_quantum=True
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    
    # 3. Training Loop
    best_auc = 0.0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for x_batch, edge_index_batch, batch_vec, pocket_batch, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x_batch = x_batch.to(DEVICE)
            edge_index_batch = edge_index_batch.to(DEVICE)
            batch_vec = batch_vec.to(DEVICE)
            pocket_batch = pocket_batch.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass with new signature
            outputs = model(x_batch, edge_index_batch, batch_vec, pocket_batch).squeeze()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, np.array(all_preds) >= 0.5)
        
        # Val
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for x_batch, edge_index_batch, batch_vec, pocket_batch, labels in val_loader:
                x_batch = x_batch.to(DEVICE)
                edge_index_batch = edge_index_batch.to(DEVICE)
                batch_vec = batch_vec.to(DEVICE)
                pocket_batch = pocket_batch.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(x_batch, edge_index_batch, batch_vec, pocket_batch).squeeze()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        val_acc = accuracy_score(val_labels, np.array(val_preds) >= 0.5)
        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except:
            val_auc = 0.0
            
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f} TrainAcc={train_acc:.4f} ValAcc={val_acc:.4f} ValAUC={val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            
    print(f"Training Complete. Best AUC: {best_auc}")

if __name__ == "__main__":
    train()
