"""
Data processing module for Drug-Patient QGNN Pipeline.

This module provides:
- PatientFeatures: Data class for patient clinical/genetic information
- DrugPatientInteraction: Data class for drug-patient interaction records
- BipartiteGraph: Bipartite graph representation with drug and patient nodes
- DrugPatientDataProcessor: Main data loading and processing class
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path


@dataclass
class PatientFeatures:
    """Patient clinical and genetic features.

    Attributes:
        patient_id: Unique patient identifier
        age: Patient age (18-100 years)
        sex: Binary sex (0=Female, 1=Male)
        weight: Body weight in kg
        genetic_markers: Array of genetic markers (SNPs, gene expression)
        comorbidities: Binary array for comorbidities (diabetes, hypertension, etc.)
        lab_values: Array of lab test results (glucose, creatinine, etc.)
        prior_medications: Array of medication history indicators
    """
    patient_id: str
    age: float
    sex: int
    weight: float
    genetic_markers: np.ndarray
    comorbidities: np.ndarray
    lab_values: np.ndarray
    prior_medications: np.ndarray

    def to_vector(self) -> np.ndarray:
        """Convert patient features to a single feature vector."""
        return np.concatenate([
            [self.age, self.sex, self.weight],
            self.genetic_markers.flatten(),
            self.comorbidities.flatten(),
            self.lab_values.flatten(),
            self.prior_medications.flatten()
        ])


@dataclass
class DrugPatientInteraction:
    """Drug-patient interaction record.

    Attributes:
        drug_id: Drug/ligand identifier
        patient_id: Patient identifier
        efficacy: Treatment efficacy (0-1 scale)
        adverse_events: List of adverse event names
        dose: Dose in mg
        duration: Treatment duration in days
        outcome: Binary outcome (0=failure, 1=success)
    """
    drug_id: str
    patient_id: str
    efficacy: float
    adverse_events: List[str]
    dose: float
    duration: float
    outcome: int

    def to_edge_features(self) -> np.ndarray:
        """Convert interaction to edge feature vector."""
        return np.array([
            self.efficacy,
            self.dose,
            self.duration,
            float(self.outcome),
            float(len(self.adverse_events))
        ])


class BipartiteGraph:
    """Bipartite graph structure for drug-patient interactions.

    The graph consists of:
    - Drug nodes with molecular features
    - Patient nodes with clinical features
    - Edges representing interactions with features

    Attributes:
        drug_nodes: Dict mapping drug_id to feature vector
        patient_nodes: Dict mapping patient_id to PatientFeatures
        edges: List of (drug_id, patient_id, edge_features) tuples
        drug_id_to_idx: Mapping from drug_id to node index
        patient_id_to_idx: Mapping from patient_id to node index
    """

    def __init__(self):
        self.drug_nodes: Dict[str, np.ndarray] = {}
        self.patient_nodes: Dict[str, PatientFeatures] = {}
        self.edges: List[Tuple[str, str, np.ndarray]] = []
        self.drug_id_to_idx: Dict[str, int] = {}
        self.patient_id_to_idx: Dict[str, int] = {}

    def add_drug(self, drug_id: str, features: np.ndarray):
        """Add a drug node to the graph."""
        if drug_id not in self.drug_nodes:
            self.drug_id_to_idx[drug_id] = len(self.drug_nodes)
            self.drug_nodes[drug_id] = features

    def add_patient(self, patient: PatientFeatures):
        """Add a patient node to the graph."""
        if patient.patient_id not in self.patient_nodes:
            self.patient_id_to_idx[patient.patient_id] = len(self.patient_nodes)
            self.patient_nodes[patient.patient_id] = patient

    def add_edge(self, drug_id: str, patient_id: str, edge_features: np.ndarray):
        """Add an edge (interaction) between drug and patient."""
        if drug_id not in self.drug_nodes:
            raise ValueError(f"Drug {drug_id} not found in graph")
        if patient_id not in self.patient_nodes:
            raise ValueError(f"Patient {patient_id} not found in graph")
        self.edges.append((drug_id, patient_id, edge_features))

    def get_drug_features_matrix(self) -> np.ndarray:
        """Get drug node features as matrix (num_drugs, drug_dim)."""
        drug_ids = sorted(self.drug_nodes.keys(), key=lambda x: self.drug_id_to_idx[x])
        return np.stack([self.drug_nodes[drug_id] for drug_id in drug_ids])

    def get_patient_features_matrix(self) -> np.ndarray:
        """Get patient node features as matrix (num_patients, patient_dim)."""
        patient_ids = sorted(self.patient_nodes.keys(), key=lambda x: self.patient_id_to_idx[x])
        return np.stack([self.patient_nodes[pid].to_vector() for pid in patient_ids])

    def get_edge_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get edge index in PyTorch Geometric format.

        Returns:
            Tuple of (edge_index, edge_features) where:
            - edge_index: (2, num_edges) array of [drug_idx, patient_idx]
            - edge_features: (num_edges, edge_dim) array
        """
        if len(self.edges) == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 5))

        edge_list = []
        edge_features = []

        for drug_id, patient_id, features in self.edges:
            drug_idx = self.drug_id_to_idx[drug_id]
            patient_idx = self.patient_id_to_idx[patient_id]
            edge_list.append([drug_idx, patient_idx])
            edge_features.append(features)

        edge_index = np.array(edge_list).T  # (2, num_edges)
        edge_features = np.stack(edge_features)  # (num_edges, edge_dim)

        return edge_index, edge_features

    def get_edge_labels(self) -> np.ndarray:
        """Get edge outcome labels (0 or 1)."""
        return np.array([features[3] for _, _, features in self.edges])

    def num_drugs(self) -> int:
        """Number of drug nodes."""
        return len(self.drug_nodes)

    def num_patients(self) -> int:
        """Number of patient nodes."""
        return len(self.patient_nodes)

    def num_edges(self) -> int:
        """Number of edges (interactions)."""
        return len(self.edges)


class DrugPatientDataProcessor:
    """Main data processor for drug-patient interaction data.

    This class handles:
    - Loading 3D molecular descriptors from PDB data
    - Creating/loading patient data
    - Creating/loading interaction data
    - Building bipartite graph representation

    Args:
        data_dir: Path to directory containing PDB data (default: othercode/data)
        seed: Random seed for reproducibility
    """

    def __init__(self, data_dir: str = "/media/priyanshu/SD/othercode/data", seed: int = 42):
        self.data_dir = data_dir
        self.seed = seed
        self.graph = BipartiteGraph()
        self._rng = np.random.RandomState(seed)

        # Track loaded data
        self._drug_descriptors_loaded = False
        self._patients_loaded = False

    def load_protein_ligand_data(self, max_samples: Optional[int] = None) -> int:
        """Load 3D molecular descriptors from PDB CSV files.

        Searches for files matching pattern:
        {data_dir}/**/results/**/*_descriptors_3d.csv

        Args:
            max_samples: Maximum number of drug samples to load (None = all)

        Returns:
            Number of drug nodes loaded
        """
        # Search for descriptor CSV files (including subdirectories under results/)
        pattern = os.path.join(self.data_dir, "**", "results", "**", "*_descriptors_3d.csv")
        csv_files = glob.glob(pattern, recursive=True)

        # Filter out hidden files (starting with .)
        csv_files = [f for f in csv_files if not os.path.basename(f).startswith('.')]

        if len(csv_files) == 0:
            print(f"Warning: No descriptor CSV files found in {self.data_dir}")
            print(f"Searched pattern: {pattern}")
            return 0

        print(f"Found {len(csv_files)} descriptor files")

        drug_count = 0
        for csv_file in csv_files[:max_samples] if max_samples else csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Extract 3D descriptor columns
                descriptor_cols = [
                    'Volume', 'PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2',
                    'Rgyr', 'Asphericity', 'SpherocityIndex', 'Eccentricity',
                    'InertialShapeFactor'
                ]

                # Add atom type counts if available
                atom_cols = ['CZ', 'CA', 'O', 'OD1', 'OG', 'N', 'NZ', 'DU']
                available_atom_cols = [col for col in atom_cols if col in df.columns]

                all_cols = descriptor_cols + available_atom_cols
                available_cols = [col for col in all_cols if col in df.columns]

                if len(available_cols) == 0:
                    print(f"Warning: No valid descriptor columns in {csv_file}")
                    continue

                # Extract features (use first row if multiple)
                features = df[available_cols].iloc[0].values.astype(np.float32)

                # Handle missing values
                features = np.nan_to_num(features, nan=0.0)

                # Create drug ID from filename
                drug_id = Path(csv_file).stem  # e.g., "1a0n--A--P27986__Repair-H_descriptors_3d"

                self.graph.add_drug(drug_id, features)
                drug_count += 1

            except Exception as e:
                print(f"Warning: Failed to load {csv_file}: {e}")
                continue

        self._drug_descriptors_loaded = True
        print(f"Loaded {drug_count} drug nodes")
        return drug_count

    def create_synthetic_patient_data(
        self,
        n_patients: int = 200,
        n_genetic_markers: int = 10,
        n_comorbidities: int = 5,
        n_lab_values: int = 8,
        n_medications: int = 15
    ):
        """Generate synthetic patient data for testing.

        Generates realistic synthetic patient features with:
        - Age: Normal(55, 15) clipped to [18, 100]
        - Sex: Bernoulli(0.5)
        - Weight: Normal(75, 15) clipped to [40, 150]
        - Genetic markers: Uniform[0, 1]
        - Comorbidities: Bernoulli(0.3)
        - Lab values: Normal distributions with realistic ranges
        - Medications: Bernoulli(0.2)

        Args:
            n_patients: Number of synthetic patients
            n_genetic_markers: Dimension of genetic marker vector
            n_comorbidities: Number of comorbidity flags
            n_lab_values: Number of lab test values
            n_medications: Number of medication history flags
        """
        print(f"Generating {n_patients} synthetic patients...")

        for i in range(n_patients):
            patient_id = f"P{i:05d}"

            # Demographics
            age = np.clip(self._rng.normal(55, 15), 18, 100)
            sex = self._rng.binomial(1, 0.5)
            weight = np.clip(self._rng.normal(75, 15), 40, 150)

            # Genetic markers (normalized to [0, 1])
            genetic_markers = self._rng.uniform(0, 1, n_genetic_markers)

            # Comorbidities (binary indicators)
            comorbidities = self._rng.binomial(1, 0.3, n_comorbidities)

            # Lab values (realistic ranges)
            lab_values = np.array([
                self._rng.normal(100, 20),    # Glucose (mg/dL)
                self._rng.normal(1.0, 0.3),   # Creatinine (mg/dL)
                self._rng.normal(30, 10),     # AST (U/L)
                self._rng.normal(30, 10),     # ALT (U/L)
                self._rng.normal(120, 30),    # LDL (mg/dL)
                self._rng.normal(50, 15),     # HDL (mg/dL)
                self._rng.normal(150, 50),    # Triglycerides (mg/dL)
                self._rng.normal(14, 2),      # Hemoglobin (g/dL)
            ])[:n_lab_values]

            # Prior medications (binary indicators)
            prior_medications = self._rng.binomial(1, 0.2, n_medications)

            patient = PatientFeatures(
                patient_id=patient_id,
                age=age,
                sex=sex,
                weight=weight,
                genetic_markers=genetic_markers,
                comorbidities=comorbidities,
                lab_values=lab_values,
                prior_medications=prior_medications
            )

            self.graph.add_patient(patient)

        self._patients_loaded = True
        print(f"Generated {n_patients} patient nodes")

    def create_synthetic_interactions(
        self,
        interaction_rate: float = 0.05,
        success_rate: float = 0.6
    ):
        """Generate synthetic drug-patient interactions.

        Creates random interactions between existing drugs and patients with
        synthetic efficacy, dose, duration, and outcome values.

        Args:
            interaction_rate: Fraction of possible drug-patient pairs to create
            success_rate: Base probability of successful outcome
        """
        if self.graph.num_drugs() == 0:
            raise ValueError("No drugs loaded. Call load_protein_ligand_data() first.")
        if self.graph.num_patients() == 0:
            raise ValueError("No patients loaded. Call create_synthetic_patient_data() first.")

        drug_ids = list(self.graph.drug_nodes.keys())
        patient_ids = list(self.graph.patient_nodes.keys())

        n_possible = len(drug_ids) * len(patient_ids)
        n_interactions = int(n_possible * interaction_rate)

        print(f"Generating {n_interactions} synthetic interactions...")

        # Sample random drug-patient pairs
        for _ in range(n_interactions):
            drug_id = self._rng.choice(drug_ids)
            patient_id = self._rng.choice(patient_ids)

            # Generate interaction features
            efficacy = self._rng.beta(2, 2)  # Beta distribution in [0, 1]
            dose = self._rng.lognormal(4, 0.5)  # Log-normal dose distribution
            duration = self._rng.gamma(5, 3)  # Gamma distribution for duration

            # Outcome depends on efficacy (with noise)
            outcome_prob = success_rate * efficacy + self._rng.normal(0, 0.1)
            outcome_prob = np.clip(outcome_prob, 0, 1)
            outcome = int(self._rng.binomial(1, outcome_prob))

            # Adverse events (more likely with higher dose)
            n_adverse = self._rng.poisson(dose / 200)
            adverse_events = [f"adverse_event_{j}" for j in range(n_adverse)]

            edge_features = np.array([
                efficacy,
                dose,
                duration,
                float(outcome),
                float(len(adverse_events))
            ])

            self.graph.add_edge(drug_id, patient_id, edge_features)

        print(f"Created {self.graph.num_edges()} interaction edges")

    def save_graph(self, filepath: str):
        """Save the bipartite graph to file using pickle.

        Args:
            filepath: Path to save the graph
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"Graph saved to {filepath}")

    def load_graph(self, filepath: str):
        """Load a bipartite graph from file.

        Args:
            filepath: Path to load the graph from
        """
        with open(filepath, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"Graph loaded from {filepath}")
        print(f"  Drugs: {self.graph.num_drugs()}")
        print(f"  Patients: {self.graph.num_patients()}")
        print(f"  Interactions: {self.graph.num_edges()}")

    def get_statistics(self) -> Dict:
        """Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_drugs': self.graph.num_drugs(),
            'num_patients': self.graph.num_patients(),
            'num_interactions': self.graph.num_edges(),
        }

        if self.graph.num_edges() > 0:
            labels = self.graph.get_edge_labels()
            stats['positive_rate'] = labels.mean()
            stats['negative_rate'] = 1 - labels.mean()

        if self.graph.num_drugs() > 0:
            drug_features = self.graph.get_drug_features_matrix()
            stats['drug_feature_dim'] = drug_features.shape[1]

        if self.graph.num_patients() > 0:
            patient_features = self.graph.get_patient_features_matrix()
            stats['patient_feature_dim'] = patient_features.shape[1]

        return stats
