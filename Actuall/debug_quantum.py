import torch
import torch.nn as nn
import numpy as np
from drug_patient_qgnn import QuantumDrugPatientGNN, DrugPatientDataProcessor
from torch.utils.data import DataLoader, Dataset

# Mock Dataset
class MockDataset(Dataset):
    def __init__(self, num_samples=100, drug_dim=11, patient_dim=19):
        self.drug_features = torch.randn(num_samples, drug_dim)
        # Simulate large values for MW (index 0)
        self.drug_features[:, 0] = self.drug_features[:, 0] * 100 + 300
        self.patient_features = torch.randn(num_samples, patient_dim)
        self.labels = torch.randint(0, 2, (num_samples,)).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.drug_features[idx], self.patient_features[idx], self.labels[idx]

def debug_training():
    print("Debugging Quantum Model Training...")
    
    # Setup
    drug_dim = 11
    patient_dim = 19
    num_qubits = 6
    device = torch.device('cpu') # Debug on CPU
    
    model = QuantumDrugPatientGNN(
        drug_dim=drug_dim,
        patient_dim=patient_dim,
        num_qubits=num_qubits,
        use_quantum=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    
    dataset = MockDataset()
    loader = DataLoader(dataset, batch_size=16)
    
    print("\nChecking Feature Stats:")
    d, p, l = next(iter(loader))
    print(f"Drug Features Mean: {d.mean(0)}")
    print(f"Drug Features Std: {d.std(0)}")
    print(f"Max Drug Feature: {d.max()}")
    
    print("\nRunning Training Steps:")
    model.train()
    for i, (drug_feats, patient_feats, labels) in enumerate(loader):
        if i >= 3: break
        
        optimizer.zero_grad()
        outputs = model(drug_feats, patient_feats).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        
        print(f"Step {i}: Loss = {loss.item():.4f}")
        
        # Check gradients
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"  Gradient Norm: {total_norm:.4f}")
        
        optimizer.step()

if __name__ == "__main__":
    debug_training()
