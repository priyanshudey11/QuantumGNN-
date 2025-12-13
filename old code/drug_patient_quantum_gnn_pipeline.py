"""
Drug-Patient Interaction Pipeline using Quantum GNN

This pipeline combines drug molecular features with patient characteristics
to predict drug-patient interactions using quantum graph neural networks.

Architecture:
1. Drug Representation: Molecular graphs (atoms as nodes, bonds as edges)
2. Patient Representation: Feature vectors (demographics, genetics, clinical data)
3. Interaction Graph: Bipartite graph connecting drugs to patients
4. Quantum GNN: Processes interaction patterns using quantum circuits

Date: 2025-11-03
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import pickle

# PennyLane for quantum circuits
try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    print("Warning: PennyLane not installed. Install with: pip install pennylane")


@dataclass
class DrugFeatures:
    """Drug molecular features"""
    drug_id: str
    smiles: str
    atoms: List[int]  # Atom types
    distances: np.ndarray  # Distance matrix
    descriptors: Dict[str, float]  # Molecular descriptors (MW, logP, etc.)
    binding_pocket_features: Optional[np.ndarray] = None  # From your PDB data


@dataclass
class PatientFeatures:
    """Patient characteristics"""
    patient_id: str
    age: float
    sex: int  # 0: Female, 1: Male
    weight: float
    genetic_markers: np.ndarray  # SNPs, gene expression, etc.
    comorbidities: np.ndarray  # Binary vector of conditions
    lab_values: np.ndarray  # Blood tests, biomarkers
    prior_medications: np.ndarray  # History of drug usage


@dataclass
class DrugPatientInteraction:
    """Drug-patient interaction data"""
    drug_id: str
    patient_id: str
    efficacy: float  # 0-1 scale or continuous
    adverse_events: List[str]  # List of side effects
    dose: float
    duration: float  # Treatment duration
    outcome: int  # Binary: success (1) or failure (0)


class DrugPatientGraph:
    """
    Bipartite graph representation for drug-patient interactions

    Graph structure:
    - Drug nodes: Molecular features from your PDB/descriptor data
    - Patient nodes: Clinical and demographic features
    - Edges: Observed or potential drug-patient interactions
    """

    def __init__(self):
        self.drugs: Dict[str, DrugFeatures] = {}
        self.patients: Dict[str, PatientFeatures] = {}
        self.interactions: List[DrugPatientInteraction] = {}

        # Graph adjacency
        self.drug_patient_edges = []  # List of (drug_idx, patient_idx)
        self.edge_features = []  # Features for each edge

    def add_drug(self, drug: DrugFeatures):
        """Add drug node to graph"""
        self.drugs[drug.drug_id] = drug

    def add_patient(self, patient: PatientFeatures):
        """Add patient node to graph"""
        self.patients[patient.patient_id] = patient

    def add_interaction(self, interaction: DrugPatientInteraction):
        """Add drug-patient interaction edge"""
        if interaction.drug_id not in self.drugs:
            raise ValueError(f"Drug {interaction.drug_id} not in graph")
        if interaction.patient_id not in self.patients:
            raise ValueError(f"Patient {interaction.patient_id} not in graph")

        drug_idx = list(self.drugs.keys()).index(interaction.drug_id)
        patient_idx = list(self.patients.keys()).index(interaction.patient_id)

        self.drug_patient_edges.append((drug_idx, patient_idx))
        self.edge_features.append({
            'efficacy': interaction.efficacy,
            'dose': interaction.dose,
            'duration': interaction.duration,
            'outcome': interaction.outcome
        })

        self.interactions[(interaction.drug_id, interaction.patient_id)] = interaction

    def get_drug_features_matrix(self) -> torch.Tensor:
        """Get drug feature matrix (N_drugs x D_drug)"""
        features = []
        for drug in self.drugs.values():
            # Combine molecular descriptors and binding pocket features
            drug_vec = []

            # Molecular descriptors
            if drug.descriptors:
                drug_vec.extend(list(drug.descriptors.values()))

            # Binding pocket features (from your CSV data)
            if drug.binding_pocket_features is not None:
                drug_vec.extend(drug.binding_pocket_features.flatten().tolist())

            features.append(drug_vec)

        # Pad to same length
        max_len = max(len(f) for f in features)
        features_padded = [f + [0.0] * (max_len - len(f)) for f in features]

        return torch.tensor(features_padded, dtype=torch.float32)

    def get_patient_features_matrix(self) -> torch.Tensor:
        """Get patient feature matrix (N_patients x D_patient)"""
        features = []
        for patient in self.patients.values():
            patient_vec = [
                patient.age,
                patient.sex,
                patient.weight
            ]
            patient_vec.extend(patient.genetic_markers.tolist())
            patient_vec.extend(patient.comorbidities.tolist())
            patient_vec.extend(patient.lab_values.tolist())
            patient_vec.extend(patient.prior_medications.tolist())

            features.append(patient_vec)

        # Pad to same length
        max_len = max(len(f) for f in features)
        features_padded = [f + [0.0] * (max_len - len(f)) for f in features]

        return torch.tensor(features_padded, dtype=torch.float32)

    def get_adjacency_matrix(self) -> torch.Tensor:
        """Get adjacency matrix for drug-patient edges"""
        n_drugs = len(self.drugs)
        n_patients = len(self.patients)
        adj = torch.zeros((n_drugs, n_patients))

        for drug_idx, patient_idx in self.drug_patient_edges:
            adj[drug_idx, patient_idx] = 1.0

        return adj

    def get_edge_labels(self) -> torch.Tensor:
        """Get interaction outcomes as labels"""
        labels = []
        for edge_feat in self.edge_features:
            labels.append(edge_feat['outcome'])
        return torch.tensor(labels, dtype=torch.float32)


class QuantumDrugPatientGNN(nn.Module):
    """
    Quantum Graph Neural Network for Drug-Patient Interaction Prediction

    Architecture:
    1. Drug Encoder: Classical embedding of molecular features
    2. Patient Encoder: Classical embedding of patient features
    3. Quantum Interaction Layer: Quantum circuit processes interaction
    4. Output Layer: Predicts interaction outcome/efficacy
    """

    def __init__(
        self,
        drug_dim: int,
        patient_dim: int,
        hidden_dim: int = 64,
        num_qubits: int = 8,
        num_qlayers: int = 2,
        use_quantum: bool = True
    ):
        super().__init__()

        self.drug_dim = drug_dim
        self.patient_dim = patient_dim
        self.hidden_dim = hidden_dim
        self.num_qubits = num_qubits
        self.num_qlayers = num_qlayers
        self.use_quantum = use_quantum and HAS_PENNYLANE

        # Drug encoder (adapts molecular features)
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_qubits)
        )

        # Patient encoder (adapts patient features)
        self.patient_encoder = nn.Sequential(
            nn.Linear(patient_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_qubits)
        )

        # Quantum layer for interaction modeling
        if self.use_quantum:
            self.quantum_device = qml.device('default.qubit', wires=num_qubits)
            self.quantum_layer = self._create_quantum_circuit()

            # Initialize quantum parameters
            weight_shapes = {"weights": (num_qlayers, num_qubits, 3)}
            self.qlayer = qml.qnn.TorchLayer(self.quantum_layer, weight_shapes)
        else:
            # Classical fallback
            self.interaction_layer = nn.Sequential(
                nn.Linear(2 * num_qubits, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_qubits)
            )

        # Output layers
        self.output_head = nn.Sequential(
            nn.Linear(num_qubits, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # For binary outcome prediction
        )

    def _create_quantum_circuit(self):
        """Create quantum circuit for drug-patient interaction"""
        @qml.qnode(self.quantum_device, interface='torch')
        def circuit(inputs, weights):
            # inputs: [drug_features, patient_features] concatenated
            # Encode drug features (first half of qubits)
            for i in range(self.num_qubits // 2):
                qml.RY(inputs[i], wires=i)

            # Encode patient features (second half of qubits)
            for i in range(self.num_qubits // 2, self.num_qubits):
                qml.RY(inputs[i], wires=i)

            # Entangle drug and patient qubits (interaction)
            for i in range(self.num_qubits // 2):
                qml.CNOT(wires=[i, i + self.num_qubits // 2])

            # Variational layers
            for layer in range(self.num_qlayers):
                for i in range(self.num_qubits):
                    qml.Rot(weights[layer, i, 0],
                           weights[layer, i, 1],
                           weights[layer, i, 2],
                           wires=i)

                # Ring topology entanglement
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.num_qubits - 1, 0])

            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        return circuit

    def forward(self, drug_features: torch.Tensor, patient_features: torch.Tensor):
        """
        Forward pass

        Args:
            drug_features: (batch_size, drug_dim)
            patient_features: (batch_size, patient_dim)

        Returns:
            predictions: (batch_size, 1) interaction probability
        """
        # Encode drug and patient features
        drug_encoded = self.drug_encoder(drug_features)  # (batch, num_qubits)
        patient_encoded = self.patient_encoder(patient_features)  # (batch, num_qubits)

        if self.use_quantum:
            # Concatenate for quantum processing
            combined = torch.cat([drug_encoded, patient_encoded], dim=1)  # (batch, 2*num_qubits)

            # Process each sample through quantum circuit
            quantum_outputs = []
            for sample in combined:
                # Normalize inputs to [0, 2π] for quantum encoding
                normalized = torch.atan(sample) * 2  # Maps to [-π, π]
                qout = self.qlayer(normalized)
                quantum_outputs.append(qout)

            quantum_output = torch.stack(quantum_outputs)  # (batch, num_qubits)
        else:
            # Classical interaction layer
            combined = torch.cat([drug_encoded, patient_encoded], dim=1)
            quantum_output = self.interaction_layer(combined)

        # Predict interaction outcome
        predictions = self.output_head(quantum_output)

        return predictions


class DrugPatientDataProcessor:
    """
    Process your existing PDB/protein-ligand data for drug-patient modeling
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.drug_descriptors = {}
        self.graph = DrugPatientGraph()

    def load_protein_ligand_data(self, max_samples: Optional[int] = None):
        """
        Load drug features from your protein-ligand binding data

        This uses the CSV files with 3D descriptors from your data
        """
        print("Loading protein-ligand binding pocket descriptors...")

        # Find all descriptor CSV files
        csv_files = list(self.data_dir.glob("**/*descriptors_3d.csv"))

        if max_samples:
            csv_files = csv_files[:max_samples]

        print(f"Found {len(csv_files)} descriptor files")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Extract pocket identifier
                if 'POCKET' in df.columns:
                    for idx, row in df.iterrows():
                        pocket_id = row['POCKET']

                        # Extract relevant features
                        feature_cols = [
                            'Volume', 'PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2',
                            'Rgyr', 'Asphericity', 'SpherocityIndex',
                            'Eccentricity', 'InertialShapeFactor',
                            'CZ', 'CA', 'O', 'N'
                        ]

                        features = {}
                        for col in feature_cols:
                            if col in df.columns:
                                features[col] = row[col]

                        self.drug_descriptors[pocket_id] = features

                        # Create DrugFeatures object
                        drug = DrugFeatures(
                            drug_id=pocket_id,
                            smiles="",  # Would need SMILES from ligand
                            atoms=[],
                            distances=np.array([]),
                            descriptors=features,
                            binding_pocket_features=row[feature_cols].values
                        )

                        self.graph.add_drug(drug)

            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

        print(f"Loaded {len(self.drug_descriptors)} drug/ligand descriptors")
        return self.drug_descriptors

    def create_synthetic_patient_data(self, n_patients: int = 100):
        """
        Create synthetic patient data for demonstration

        In practice, you would load real patient data from clinical databases
        """
        print(f"Creating {n_patients} synthetic patient profiles...")

        for i in range(n_patients):
            patient = PatientFeatures(
                patient_id=f"PATIENT_{i:04d}",
                age=np.random.uniform(18, 85),
                sex=np.random.randint(0, 2),
                weight=np.random.uniform(50, 120),
                genetic_markers=np.random.randn(10),  # 10 genetic markers
                comorbidities=np.random.randint(0, 2, 5),  # 5 common conditions
                lab_values=np.random.randn(8),  # 8 lab test values
                prior_medications=np.random.randint(0, 2, 10)  # 10 medication history
            )

            self.graph.add_patient(patient)

        print(f"Created {n_patients} patient profiles")

    def create_synthetic_interactions(self, interaction_rate: float = 0.1):
        """
        Create synthetic drug-patient interactions

        In practice, you would load real clinical trial data or EHR data
        """
        drug_ids = list(self.graph.drugs.keys())
        patient_ids = list(self.graph.patients.keys())

        n_interactions = int(len(drug_ids) * len(patient_ids) * interaction_rate)
        print(f"Creating {n_interactions} synthetic interactions...")

        for _ in range(n_interactions):
            drug_id = np.random.choice(drug_ids)
            patient_id = np.random.choice(patient_ids)

            # Skip if interaction already exists
            if (drug_id, patient_id) in self.graph.interactions:
                continue

            interaction = DrugPatientInteraction(
                drug_id=drug_id,
                patient_id=patient_id,
                efficacy=np.random.uniform(0.3, 0.95),
                adverse_events=[],
                dose=np.random.uniform(10, 500),
                duration=np.random.uniform(7, 90),
                outcome=np.random.randint(0, 2)
            )

            self.graph.add_interaction(interaction)

        print(f"Created {len(self.graph.interactions)} interactions")

    def save_graph(self, filepath: str):
        """Save processed graph to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"Saved graph to {filepath}")

    def load_graph(self, filepath: str):
        """Load processed graph from disk"""
        with open(filepath, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"Loaded graph from {filepath}")
        return self.graph


class DrugPatientTrainer:
    """
    Training pipeline for quantum drug-patient interaction model
    """

    def __init__(
        self,
        model: QuantumDrugPatientGNN,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

        self.train_losses = []
        self.val_losses = []

    def train_epoch(
        self,
        drug_features: torch.Tensor,
        patient_features: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Train for one epoch"""
        self.model.train()

        drug_features = drug_features.to(self.device)
        patient_features = patient_features.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        predictions = self.model(drug_features, patient_features)
        loss = self.criterion(predictions.squeeze(), labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(
        self,
        drug_features: torch.Tensor,
        patient_features: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()

        with torch.no_grad():
            drug_features = drug_features.to(self.device)
            patient_features = patient_features.to(self.device)
            labels = labels.to(self.device)

            predictions = self.model(drug_features, patient_features)
            loss = self.criterion(predictions.squeeze(), labels)

            # Calculate accuracy
            predicted_classes = (predictions.squeeze() > 0.5).float()
            accuracy = (predicted_classes == labels).float().mean()

        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }

    def fit(
        self,
        graph: DrugPatientGraph,
        epochs: int = 100,
        val_split: float = 0.2
    ):
        """Train model on drug-patient interaction graph"""
        print(f"Training for {epochs} epochs...")

        # Get features and labels
        drug_features = graph.get_drug_features_matrix()
        patient_features = graph.get_patient_features_matrix()

        # Create dataset for each interaction
        X_drug, X_patient, y = [], [], []
        for (drug_idx, patient_idx), edge_feat in zip(graph.drug_patient_edges, graph.edge_features):
            X_drug.append(drug_features[drug_idx])
            X_patient.append(patient_features[patient_idx])
            y.append(edge_feat['outcome'])

        X_drug = torch.stack(X_drug)
        X_patient = torch.stack(X_patient)
        y = torch.tensor(y, dtype=torch.float32)

        # Train/val split
        n_train = int(len(y) * (1 - val_split))
        indices = torch.randperm(len(y))

        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        # Training loop
        for epoch in range(epochs):
            train_loss = self.train_epoch(
                X_drug[train_idx],
                X_patient[train_idx],
                y[train_idx]
            )

            val_metrics = self.evaluate(
                X_drug[val_idx],
                X_patient[val_idx],
                y[val_idx]
            )

            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}")

        print("Training complete!")
        return self.train_losses, self.val_losses


def main_pipeline_demo():
    """
    Complete pipeline demonstration
    """
    print("=" * 60)
    print("Drug-Patient Interaction Quantum GNN Pipeline")
    print("=" * 60)

    # 1. Load and process data
    data_dir = "/Users/priyanshudey/Code/Qunatum/othercode/data"
    processor = DrugPatientDataProcessor(data_dir)

    # Load drug data from your protein-ligand descriptors
    processor.load_protein_ligand_data(max_samples=50)  # Limit for demo

    # Create synthetic patient data
    processor.create_synthetic_patient_data(n_patients=100)

    # Create synthetic interactions
    processor.create_synthetic_interactions(interaction_rate=0.05)

    # Save graph
    processor.save_graph("drug_patient_graph.pkl")

    # 2. Initialize model
    graph = processor.graph
    drug_dim = graph.get_drug_features_matrix().shape[1]
    patient_dim = graph.get_patient_features_matrix().shape[1]

    print(f"\nModel dimensions:")
    print(f"  Drug features: {drug_dim}")
    print(f"  Patient features: {patient_dim}")
    print(f"  Number of drugs: {len(graph.drugs)}")
    print(f"  Number of patients: {len(graph.patients)}")
    print(f"  Number of interactions: {len(graph.interactions)}")

    model = QuantumDrugPatientGNN(
        drug_dim=drug_dim,
        patient_dim=patient_dim,
        hidden_dim=64,
        num_qubits=8,
        num_qlayers=2,
        use_quantum=HAS_PENNYLANE
    )

    print(f"\nModel created with {'quantum' if HAS_PENNYLANE else 'classical'} layers")

    # 3. Train model
    trainer = DrugPatientTrainer(model, learning_rate=0.001)
    trainer.fit(graph, epochs=50, val_split=0.2)

    # 4. Save model
    torch.save(model.state_dict(), "quantum_drug_patient_model.pt")
    print("\nModel saved to quantum_drug_patient_model.pt")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main_pipeline_demo()