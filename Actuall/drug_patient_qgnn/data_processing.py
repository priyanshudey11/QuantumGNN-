"""
Data processing module for Drug-Protein QGNN Pipeline.

This module provides:
- ProteinPocketFeatures: Data class for protein pocket 3D descriptors
- DrugProteinInteraction: Data class for drug-protein interaction records
- BipartiteGraph: Bipartite graph representation with ligand and protein pocket nodes
- DrugProteinDataProcessor: Main data loading and processing class
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
from tqdm import tqdm


@dataclass
class ProteinPocketFeatures:
    """Protein pocket 3D geometric and atom-type features.

    Attributes:
        pocket_id: Unique pocket identifier (e.g., PDB_chain_pocket)
        pocket_type: Type of pocket (PLOC, PLONC, PLA, HD)
        volume: Pocket volume
        pmi1, pmi2, pmi3: Principal moments of inertia
        npr1, npr2: Normalized principal moments ratios
        rgyr: Radius of gyration
        asphericity: Asphericity measure
        spherocity_index: Spherocity index
        eccentricity: Eccentricity measure
        inertial_shape_factor: Inertial shape factor
        atom_type_counts: Dictionary of atom type counts (CZ, CA, O, etc.)
        additional_features: Optional additional descriptor values
    """
    pocket_id: str
    pocket_type: str = "unknown"  # PLOC, PLONC, PLA, HD
    volume: float = 0.0
    pmi1: float = 0.0
    pmi2: float = 0.0
    pmi3: float = 0.0
    npr1: float = 0.0
    npr2: float = 0.0
    rgyr: float = 0.0
    asphericity: float = 0.0
    spherocity_index: float = 0.0
    eccentricity: float = 0.0
    inertial_shape_factor: float = 0.0
    atom_type_counts: Dict[str, float] = field(default_factory=dict)
    additional_features: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_vector(self) -> np.ndarray:
        """Convert pocket features to a single feature vector."""
        # Core geometric descriptors
        geometric = np.array([
            self.volume, self.pmi1, self.pmi2, self.pmi3,
            self.npr1, self.npr2, self.rgyr, self.asphericity,
            self.spherocity_index, self.eccentricity, self.inertial_shape_factor
        ])

        # Atom type counts (in consistent order)
        atom_types = ['CZ', 'CA', 'O', 'OD1', 'OG', 'N', 'NZ', 'DU']
        atom_counts = np.array([self.atom_type_counts.get(at, 0.0) for at in atom_types])

        # Combine all features
        features = np.concatenate([geometric, atom_counts])

        if len(self.additional_features) > 0:
            features = np.concatenate([features, self.additional_features])

        return features


class SimpleMol2Parser:
    """Simple parser for MOL2 files to extract basic chemical features."""
    
    ATOM_WEIGHTS = {
        'C': 12.01, 'N': 14.01, 'O': 16.00, 'S': 32.06, 'P': 30.97,
        'F': 19.00, 'Cl': 35.45, 'Br': 79.90, 'I': 126.90, 'H': 1.008
    }
    
    @staticmethod
    def parse(filepath: str) -> 'LigandFeatures':
        """Parse a MOL2 file and return LigandFeatures."""
        atom_counts = {atom: 0 for atom in SimpleMol2Parser.ATOM_WEIGHTS}
        mol_weight = 0.0
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            in_atom_section = False
            
            for line in lines:
                line = line.strip()
                if line == "@<TRIPOS>ATOM":
                    in_atom_section = True
                    continue
                elif line.startswith("@<TRIPOS>"):
                    in_atom_section = False
                
                if in_atom_section:
                    parts = line.split()
                    if len(parts) >= 6:
                        atom_type_full = parts[5]
                        atom_type = atom_type_full.split('.')[0] if '.' in atom_type_full else atom_type_full
                        
                        # Normalize atom type
                        if atom_type in SimpleMol2Parser.ATOM_WEIGHTS:
                            atom_counts[atom_type] += 1
                            mol_weight += SimpleMol2Parser.ATOM_WEIGHTS[atom_type]
                        elif atom_type == 'H': # Handle H explicit if present
                             atom_counts['H'] += 1
                             mol_weight += SimpleMol2Parser.ATOM_WEIGHTS['H']

        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            
        return LigandFeatures(
            ligand_id=Path(filepath).stem,
            molecular_weight=mol_weight,
            atom_counts=atom_counts
        )

@dataclass
class LigandFeatures:
    """Ligand (drug) molecular features.

    Attributes:
        ligand_id: Unique ligand identifier
        molecular_weight: Molecular weight in Da
        atom_counts: Dictionary of atom counts
        additional_features: Optional additional descriptor values
    """
    ligand_id: str
    molecular_weight: float = 0.0
    atom_counts: Dict[str, int] = field(default_factory=dict)
    additional_features: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_vector(self) -> np.ndarray:
        """Convert ligand features to a single feature vector."""
        # Atom types to include in vector (consistent order)
        atom_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']
        
        counts = np.array([self.atom_counts.get(at, 0) for at in atom_types], dtype=np.float32)
        
        features = np.concatenate([
            np.array([self.molecular_weight], dtype=np.float32),
            counts
        ])

        if len(self.additional_features) > 0:
            features = np.concatenate([features, self.additional_features])

        return features


@dataclass
class DrugProteinInteraction:
    """Drug-protein interaction record.

    Attributes:
        ligand_id: Ligand/drug identifier
        pocket_id: Protein pocket identifier
        interaction_type: Type of interaction (PLOC, PLONC, PLA, HD/negative)
        binding_affinity: Binding affinity (if available, in kcal/mol)
        distance: Distance metric (if available)
        outcome: Binary outcome (0=no binding/HD, 1=binding)
    """
    ligand_id: str
    pocket_id: str
    interaction_type: str  # PLOC, PLONC, PLA, HD
    binding_affinity: Optional[float] = None
    distance: Optional[float] = None
    outcome: int = 1  # 1 for binding (PLOC/PLONC/PLA), 0 for non-binding (HD)

    def to_edge_features(self) -> np.ndarray:
        """Convert interaction to edge feature vector."""
        return np.array([
            float(self.outcome),
            self.binding_affinity if self.binding_affinity is not None else 0.0,
            self.distance if self.distance is not None else 0.0,
            float(['HD', 'PLOC', 'PLONC', 'PLA'].index(self.interaction_type)
                  if self.interaction_type in ['HD', 'PLOC', 'PLONC', 'PLA'] else 0)
        ])


class BipartiteGraph:
    """Bipartite graph structure for drug-protein interactions.

    The graph consists of:
    - Ligand (drug) nodes with molecular features
    - Protein pocket nodes with 3D geometric features
    - Edges representing interactions with features

    Attributes:
        ligand_nodes: Dict mapping ligand_id to feature vector
        pocket_nodes: Dict mapping pocket_id to ProteinPocketFeatures
        edges: List of (ligand_id, pocket_id, edge_features) tuples
        ligand_id_to_idx: Mapping from ligand_id to node index
        pocket_id_to_idx: Mapping from pocket_id to node index
    """

    def __init__(self):
        self.ligand_nodes: Dict[str, np.ndarray] = {}
        self.pocket_nodes: Dict[str, ProteinPocketFeatures] = {}
        self.edges: List[Tuple[str, str, np.ndarray]] = []
        self.ligand_id_to_idx: Dict[str, int] = {}
        self.pocket_id_to_idx: Dict[str, int] = {}

    def add_ligand(self, ligand_id: str, features: np.ndarray):
        """Add a ligand node to the graph."""
        if ligand_id not in self.ligand_nodes:
            self.ligand_id_to_idx[ligand_id] = len(self.ligand_nodes)
            self.ligand_nodes[ligand_id] = features

    def add_pocket(self, pocket: ProteinPocketFeatures):
        """Add a protein pocket node to the graph."""
        if pocket.pocket_id not in self.pocket_nodes:
            self.pocket_id_to_idx[pocket.pocket_id] = len(self.pocket_nodes)
            self.pocket_nodes[pocket.pocket_id] = pocket

    def add_edge(self, ligand_id: str, pocket_id: str, edge_features: np.ndarray):
        """Add an edge (interaction) between ligand and pocket."""
        if ligand_id not in self.ligand_nodes:
            raise ValueError(f"Ligand {ligand_id} not found in graph")
        if pocket_id not in self.pocket_nodes:
            raise ValueError(f"Pocket {pocket_id} not found in graph")
        self.edges.append((ligand_id, pocket_id, edge_features))

    def get_ligand_features_matrix(self) -> np.ndarray:
        """Get ligand node features as matrix (num_ligands, ligand_dim)."""
        ligand_ids = sorted(self.ligand_nodes.keys(), key=lambda x: self.ligand_id_to_idx[x])
        return np.stack([self.ligand_nodes[ligand_id] for ligand_id in ligand_ids])

    def get_pocket_features_matrix(self) -> np.ndarray:
        """Get pocket node features as matrix (num_pockets, pocket_dim)."""
        pocket_ids = sorted(self.pocket_nodes.keys(), key=lambda x: self.pocket_id_to_idx[x])
        return np.stack([self.pocket_nodes[pid].to_vector() for pid in pocket_ids])

    # Backward compatibility aliases
    def get_drug_features_matrix(self) -> np.ndarray:
        """Alias for get_ligand_features_matrix() for backward compatibility."""
        return self.get_ligand_features_matrix()

    def get_patient_features_matrix(self) -> np.ndarray:
        """Alias for get_pocket_features_matrix() for backward compatibility."""
        return self.get_pocket_features_matrix()

    def get_edge_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get edge index in PyTorch Geometric format.

        Returns:
            Tuple of (edge_index, edge_features) where:
            - edge_index: (2, num_edges) array of [ligand_idx, pocket_idx]
            - edge_features: (num_edges, edge_dim) array
        """
        if len(self.edges) == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4))

        edge_list = []
        edge_features = []

        for ligand_id, pocket_id, features in self.edges:
            ligand_idx = self.ligand_id_to_idx[ligand_id]
            pocket_idx = self.pocket_id_to_idx[pocket_id]
            edge_list.append([ligand_idx, pocket_idx])
            edge_features.append(features)

        edge_index = np.array(edge_list).T  # (2, num_edges)
        edge_features = np.stack(edge_features)  # (num_edges, edge_dim)

        return edge_index, edge_features

    def get_edge_labels(self) -> np.ndarray:
        """Get edge outcome labels (0 or 1)."""
        return np.array([features[0] for _, _, features in self.edges])

    def num_ligands(self) -> int:
        """Number of ligand nodes."""
        return len(self.ligand_nodes)

    def num_pockets(self) -> int:
        """Number of protein pocket nodes."""
        return len(self.pocket_nodes)

    # Backward compatibility aliases
    def num_drugs(self) -> int:
        """Alias for num_ligands() for backward compatibility."""
        return self.num_ligands()

    def num_patients(self) -> int:
        """Alias for num_pockets() for backward compatibility."""
        return self.num_pockets()

    def num_edges(self) -> int:
        """Number of edges (interactions)."""
        return len(self.edges)


class DrugProteinDataProcessor:
    """Main data processor for drug-protein interaction data.

    This class handles:
    - Loading 3D pocket descriptors from PDB data
    - Creating/loading ligand data
    - Creating/loading interaction data
    - Building bipartite graph representation

    Args:
        data_dir: Path to directory containing PDB data (default: /media/priyanshu/SD/othercode/data)
        seed: Random seed for reproducibility
    """

    def __init__(self, data_dir: str = "/media/priyanshu/SD/othercode/data", seed: int = 42):
        self.data_dir = data_dir
        self.seed = seed
        self.graph = BipartiteGraph()
        self._rng = np.random.RandomState(seed)

        # Track loaded data
        self._pockets_loaded = False
        self._ligands_loaded = False

    def load_real_data(self, data_dir: str, max_samples: Optional[int] = None) -> Dict[str, int]:
        """
        Loads real Protein (Pocket) and Drug (Ligand) data from the filesystem.
        
        Args:
            data_dir: Base directory containing PDB results
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            Dictionary with counts of loaded nodes and edges
        """
        print(f"Searching for data in: {data_dir}")
        
        # Find all descriptor CSV files (Protein Pockets)
        pattern = os.path.join(data_dir, "**", "results", "**", "*_descriptors_3d.csv")
        csv_files = glob.glob(pattern, recursive=True)
        
        if max_samples:
            csv_files = csv_files[:max_samples]
            
        print(f"Found {len(csv_files)} protein descriptor files")
        
        loaded_proteins = 0
        loaded_drugs = 0
        interactions = 0
        
        # Track existing interactions to avoid duplicates when generating negatives
        existing_interactions = set()
        
        # Columns expected in CSV
        descriptor_cols = [
            'Volume', 'PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2',
            'Rgyr', 'Asphericity', 'SpherocityIndex', 'Eccentricity',
            'InertialShapeFactor'
        ]
        atom_cols = ['CZ', 'CA', 'O', 'OD1', 'OG', 'N', 'NZ', 'DU']
        
        # Lists to keep track of loaded IDs for negative sampling
        all_protein_ids = []
        all_drug_ids = []
        
        for csv_file in tqdm(csv_files, desc="Loading PDB Data"):
            try:
                # 1. Load Protein Pocket (from CSV)
                df = pd.read_csv(csv_file)
                if df.empty:
                    continue
                
                # Check for required columns
                if not all(col in df.columns for col in descriptor_cols):
                    continue
                    
                row = df.iloc[0]
                
                # Create ProteinPocketFeatures object
                protein_id = Path(csv_file).stem.replace("_descriptors_3d", "")
                pocket = ProteinPocketFeatures(pocket_id=protein_id)
                
                # Populate geometric features
                pocket.volume = float(row.get('Volume', 0.0))
                pocket.pmi1 = float(row.get('PMI1', 0.0))
                pocket.pmi2 = float(row.get('PMI2', 0.0))
                pocket.pmi3 = float(row.get('PMI3', 0.0))
                pocket.npr1 = float(row.get('NPR1', 0.0))
                pocket.npr2 = float(row.get('NPR2', 0.0))
                pocket.rgyr = float(row.get('Rgyr', 0.0))
                pocket.asphericity = float(row.get('Asphericity', 0.0))
                pocket.spherocity_index = float(row.get('SpherocityIndex', 0.0))
                pocket.eccentricity = float(row.get('Eccentricity', 0.0))
                pocket.inertial_shape_factor = float(row.get('InertialShapeFactor', 0.0))
                
                # Populate atom counts
                for atom_type in atom_cols:
                    if atom_type in row:
                        pocket.atom_type_counts[atom_type] = float(row[atom_type])
                
                self.graph.add_pocket(pocket)
                loaded_proteins += 1
                all_protein_ids.append(protein_id)
                
                # 2. Find associated Drugs/Ligands (MOL2 files in same dir)
                parent_dir = Path(csv_file).parent
                mol2_files = list(parent_dir.glob("*.mol2"))
                
                # Filter out hidden files
                mol2_files = [f for f in mol2_files if not f.name.startswith('.')]
                
                for mol2_file in mol2_files:
                    drug_id = mol2_file.stem
                    
                    # Create Drug Node
                    # Parse MOL2 file to get real features
                    ligand_features = SimpleMol2Parser.parse(str(mol2_file))
                    drug_features = ligand_features.to_vector()
                    
                    self.graph.add_ligand(drug_id, drug_features)
                    loaded_drugs += 1
                    all_drug_ids.append(drug_id)
                    
                    # 3. Create Interaction (Positive)
                    # In this dataset, co-location implies interaction (complex)
                    # We assume a positive interaction (label=1) for the crystal structure
                    edge_features = np.array([1.0, 0.0, 0.0, 0.0])
                    self.graph.add_edge(drug_id, protein_id, edge_features)
                    interactions += 1
                    existing_interactions.add((drug_id, protein_id))
                    
            except Exception as e:
                # print(f"Error loading {csv_file}: {e}")
                continue
        
        # 4. Generate Negative Interactions (Optimized)
        # We aim for a 1:1 ratio of positive to negative samples
        print(f"Generating negative samples (target: {interactions})...")
        negatives_generated = 0
        
        if loaded_proteins > 0 and loaded_drugs > 0:
            # Convert lists to numpy arrays for faster sampling
            all_drug_ids_np = np.array(all_drug_ids)
            all_protein_ids_np = np.array(all_protein_ids)
            
            # Generate candidate pairs in batches
            batch_size = 10000
            
            with tqdm(total=interactions, desc="Generating Negatives") as pbar:
                while negatives_generated < interactions:
                    # Generate random indices
                    drug_indices = self._rng.randint(0, len(all_drug_ids_np), size=batch_size)
                    protein_indices = self._rng.randint(0, len(all_protein_ids_np), size=batch_size)
                    
                    batch_drugs = all_drug_ids_np[drug_indices]
                    batch_proteins = all_protein_ids_np[protein_indices]
                    
                    for d_id, p_id in zip(batch_drugs, batch_proteins):
                        if negatives_generated >= interactions:
                            break
                            
                        # Check if interaction already exists
                        if (d_id, p_id) not in existing_interactions:
                            # Add negative interaction (label=0.0)
                            edge_features = np.array([0.0, 0.0, 0.0, 0.0])
                            self.graph.add_edge(d_id, p_id, edge_features)
                            existing_interactions.add((d_id, p_id))
                            
                            negatives_generated += 1
                            pbar.update(1)
            
        self._pockets_loaded = True
        self._ligands_loaded = True
        
        total_interactions = interactions + negatives_generated
        print(f"Loaded {loaded_proteins} proteins, {loaded_drugs} drugs")
        print(f"Interactions: {interactions} positive, {negatives_generated} negative (Total: {total_interactions})")
        
        return {
            "proteins": loaded_proteins,
            "drugs": loaded_drugs,
            "interactions": total_interactions,
            "positives": interactions,
            "negatives": negatives_generated
        }

    def load_protein_ligand_data(self, *args, **kwargs):
        """Deprecated method."""
        raise RuntimeError("This method is deprecated. Use load_real_data() instead.")

    def load_drug_data_from_pdb(self, *args, **kwargs):
        """Deprecated method."""
        raise RuntimeError("This method is deprecated. Use load_real_data() instead.")

    def create_synthetic_ligand_data(self, *args, **kwargs):
        """Deprecated method."""
        raise RuntimeError("This method is deprecated. Use load_real_data() instead.")
        
    def create_synthetic_patient_data(self, *args, **kwargs):
        """Deprecated method."""
        raise RuntimeError("This method is deprecated. Use load_real_data() instead.")
    
    def create_synthetic_interactions(self, *args, **kwargs):
        """Deprecated method."""
        raise RuntimeError("This method is deprecated. Use load_real_data() instead.")

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
        print(f"  Ligands: {self.graph.num_ligands()}")
        print(f"  Pockets: {self.graph.num_pockets()}")
        print(f"  Interactions: {self.graph.num_edges()}")

    def get_statistics(self) -> Dict:
        """Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_ligands': self.graph.num_ligands(),
            'num_pockets': self.graph.num_pockets(),
            'num_interactions': self.graph.num_edges(),
            # Backward compatibility
            'num_drugs': self.graph.num_ligands(),
            'num_patients': self.graph.num_pockets(),
        }

        if self.graph.num_edges() > 0:
            labels = self.graph.get_edge_labels()
            stats['positive_rate'] = labels.mean()
            stats['negative_rate'] = 1 - labels.mean()

        if self.graph.num_ligands() > 0:
            ligand_features = self.graph.get_ligand_features_matrix()
            stats['ligand_feature_dim'] = ligand_features.shape[1]
            stats['drug_feature_dim'] = ligand_features.shape[1]  # Backward compatibility

        if self.graph.num_pockets() > 0:
            pocket_features = self.graph.get_pocket_features_matrix()
            stats['pocket_feature_dim'] = pocket_features.shape[1]
            stats['patient_feature_dim'] = pocket_features.shape[1]  # Backward compatibility

        return stats


# Backward compatibility aliases
PatientFeatures = ProteinPocketFeatures
DrugPatientInteraction = DrugProteinInteraction
DrugPatientDataProcessor = DrugProteinDataProcessor
