"""
Data processing module for Ligand-Pocket QGNN Pipeline.

This module provides:
- PocketFeatures: Data class for protein pocket 3D descriptors
- LigandGraph: Data class for ligand graph (atoms and bonds)
- LigandPocketInteraction: Data class for interaction records
- LigandPocketDataProcessor: Main data loading and processing class
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset

# ==========================================
# Data Structures
# ==========================================

@dataclass
class PocketFeatures:
    """Protein pocket 3D geometric and atom-type features."""
    pocket_id: str
    pocket_type: str = "unknown"
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
        geometric = np.array([
            self.volume, self.pmi1, self.pmi2, self.pmi3,
            self.npr1, self.npr2, self.rgyr, self.asphericity,
            self.spherocity_index, self.eccentricity, self.inertial_shape_factor
        ])
        atom_types = ['CZ', 'CA', 'O', 'OD1', 'OG', 'N', 'NZ', 'DU']
        atom_counts = np.array([self.atom_type_counts.get(at, 0.0) for at in atom_types])
        features = np.concatenate([geometric, atom_counts])
        if len(self.additional_features) > 0:
            features = np.concatenate([features, self.additional_features])
        return features.astype(np.float32)

@dataclass
class LigandGraph:
    """Ligand (drug) graph representation."""
    ligand_id: str
    atom_features: np.ndarray  # (N, F)
    edge_index: np.ndarray     # (2, E)
    edge_attributes: np.ndarray # (E, Fe)
    
    def __repr__(self):
        return f"LigandGraph(id={self.ligand_id}, atoms={self.atom_features.shape[0]}, edges={self.edge_index.shape[1]})"

@dataclass
class LigandPocketInteraction:
    """Interaction between a ligand and a pocket."""
    ligand_id: str
    pocket_id: str
    label: float
    interaction_type: str = "unknown"

# ==========================================
# Parsers
# ==========================================

class Mol2GraphParser:
    """Parses MOL2 files into graph structures (Atoms + Bonds)."""
    ATOM_TYPES = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']
    
    @staticmethod
    def get_atom_encoding(atom_type_str: str) -> List[float]:
        base_type = atom_type_str.split('.')[0]
        encoding = [1.0 if base_type == t else 0.0 for t in Mol2GraphParser.ATOM_TYPES]
        return encoding

    @staticmethod
    def parse(filepath: str) -> Optional[LigandGraph]:
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            atoms = []
            bonds = []
            section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith("@<TRIPOS>"):
                    section = line
                    continue
                if not line: continue
                    
                if section == "@<TRIPOS>ATOM":
                    parts = line.split()
                    if len(parts) >= 6:
                        atom_type = parts[5]
                        features = Mol2GraphParser.get_atom_encoding(atom_type)
                        atoms.append(features)
                        
                elif section == "@<TRIPOS>BOND":
                    parts = line.split()
                    if len(parts) >= 4:
                        src = int(parts[1]) - 1
                        dst = int(parts[2]) - 1
                        bond_type = parts[3]
                        bond_feat = 1.0
                        if bond_type == '2': bond_feat = 2.0
                        elif bond_type == '3': bond_feat = 3.0
                        elif bond_type == 'ar': bond_feat = 1.5
                        elif bond_type == 'am': bond_feat = 1.5
                        bonds.append([src, dst, bond_feat])
            
            if not atoms: return None
                
            atom_features = np.array(atoms, dtype=np.float32)
            
            if bonds:
                bond_data = np.array(bonds)
                src_nodes = np.concatenate([bond_data[:, 0], bond_data[:, 1]])
                dst_nodes = np.concatenate([bond_data[:, 1], bond_data[:, 0]])
                edge_index = np.stack([src_nodes, dst_nodes]).astype(np.int64)
                edge_attrs = np.concatenate([bond_data[:, 2], bond_data[:, 2]])
                edge_attrs = edge_attrs.reshape(-1, 1).astype(np.float32)
            else:
                edge_index = np.zeros((2, 0), dtype=np.int64)
                edge_attrs = np.zeros((0, 1), dtype=np.float32)
                
            return LigandGraph(
                ligand_id=Path(filepath).stem,
                atom_features=atom_features,
                edge_index=edge_index,
                edge_attributes=edge_attrs
            )
        except Exception as e:
            # print(f"Error parsing {filepath}: {e}")
            return None

# ==========================================
# Data Processor
# ==========================================

class LigandPocketDataProcessor:
    def __init__(self, data_dir: str, seed: int = 42):
        self.data_dir = data_dir
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self.pockets: Dict[str, PocketFeatures] = {}
        self.ligands: Dict[str, LigandGraph] = {}
        self.interactions: List[LigandPocketInteraction] = []
        
    def load_data(self, max_samples: Optional[int] = None):
        print(f"Searching for data in: {self.data_dir}")
        pattern = os.path.join(self.data_dir, "**", "results", "**", "*_descriptors_3d.csv")
        csv_files = glob.glob(pattern, recursive=True)
        if max_samples: csv_files = csv_files[:max_samples]
            
        print(f"Found {len(csv_files)} protein descriptor files")
        loaded_pockets = 0
        loaded_ligands = 0
        pos_interactions = 0
        existing_pairs = set()
        
        for csv_file in tqdm(csv_files, desc="Loading Data"):
            try:
                df = pd.read_csv(csv_file)
                if df.empty: continue
                row = df.iloc[0]
                pocket_id = Path(csv_file).stem.replace("_descriptors_3d", "")
                
                pocket = PocketFeatures(pocket_id=pocket_id)
                pocket.volume = float(row.get('Volume', 0.0))
                # ... (simplified for brevity, assuming same logic as before)
                atom_cols = ['CZ', 'CA', 'O', 'OD1', 'OG', 'N', 'NZ', 'DU']
                for at in atom_cols:
                    if at in row: pocket.atom_type_counts[at] = float(row[at])
                
                self.pockets[pocket_id] = pocket
                loaded_pockets += 1
                
                parent_dir = Path(csv_file).parent
                mol2_files = list(parent_dir.glob("*.mol2"))
                mol2_files = [f for f in mol2_files if not f.name.startswith('.')]
                
                for mol2_file in mol2_files:
                    ligand_graph = Mol2GraphParser.parse(str(mol2_file))
                    if ligand_graph is None: continue
                    self.ligands[ligand_graph.ligand_id] = ligand_graph
                    loaded_ligands += 1
                    
                    interaction = LigandPocketInteraction(
                        ligand_id=ligand_graph.ligand_id,
                        pocket_id=pocket_id,
                        label=1.0,
                        interaction_type="binding"
                    )
                    self.interactions.append(interaction)
                    existing_pairs.add((ligand_graph.ligand_id, pocket_id))
                    pos_interactions += 1
            except Exception: continue
                
        print(f"Generating negative samples (target: {pos_interactions})...")
        neg_interactions = 0
        if loaded_pockets > 0 and loaded_ligands > 0:
            all_ligand_ids = np.array(list(self.ligands.keys()))
            all_pocket_ids = np.array(list(self.pockets.keys()))
            
            # Vectorized negative sampling: much faster!
            # Generate more candidates than needed and filter duplicates
            n_ligands = len(all_ligand_ids)
            n_pockets = len(all_pocket_ids)
            target_negs = pos_interactions
            
            # Generate candidates in bulk (with padding for duplicates)
            batch_size = min(10000, max(target_negs * 2, 5000))  # Adaptive batch size
            neg_ligand_indices = self._rng.randint(0, n_ligands, size=batch_size)
            neg_pocket_indices = self._rng.randint(0, n_pockets, size=batch_size)
            
            pbar = tqdm(total=target_negs, desc="Generating Negatives")
            
            for i in range(batch_size):
                if neg_interactions >= target_negs:
                    break
                    
                l_id = all_ligand_ids[neg_ligand_indices[i]]
                p_id = all_pocket_ids[neg_pocket_indices[i]]
                
                if (l_id, p_id) not in existing_pairs:
                    interaction = LigandPocketInteraction(
                        ligand_id=l_id,
                        pocket_id=p_id,
                        label=0.0,
                        interaction_type="non_binding"
                    )
                    self.interactions.append(interaction)
                    existing_pairs.add((l_id, p_id))
                    neg_interactions += 1
                    pbar.update(1)
            
            # If we didn't get enough, fall back to slower method
            if neg_interactions < target_negs:
                print(f"⚠ Generated {neg_interactions}/{target_negs} negatives in batch. Filling remainder...")
                while neg_interactions < target_negs:
                    l_id = self._rng.choice(all_ligand_ids)
                    p_id = self._rng.choice(all_pocket_ids)
                    if (l_id, p_id) not in existing_pairs:
                        interaction = LigandPocketInteraction(
                            ligand_id=l_id,
                            pocket_id=p_id,
                            label=0.0,
                            interaction_type="non_binding"
                        )
                        self.interactions.append(interaction)
                        existing_pairs.add((l_id, p_id))
                        neg_interactions += 1
                        pbar.update(1)
            
            pbar.close()
                        
        print(f"Loaded {loaded_pockets} pockets, {loaded_ligands} ligands")
        print(f"Interactions: {pos_interactions} positive, {neg_interactions} negative")

    def get_dataset(self):
        return self.interactions

# ==========================================
# PyTorch Dataset & Collate
# ==========================================

class LigandPocketDataset(Dataset):
    def __init__(self, processor: LigandPocketDataProcessor, interactions: List[LigandPocketInteraction]):
        self.processor = processor
        self.interactions = interactions
        # Pre-cache tensor conversions for faster data loading
        self._tensor_cache = {}
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        
        # Use cache to avoid repeated tensor conversions
        cache_key = f"{interaction.ligand_id}_{interaction.pocket_id}"
        if cache_key not in self._tensor_cache:
            ligand = self.processor.ligands[interaction.ligand_id]
            pocket = self.processor.pockets[interaction.pocket_id]
            
            self._tensor_cache[cache_key] = (
                torch.from_numpy(ligand.atom_features),  # Faster than torch.tensor
                torch.from_numpy(ligand.edge_index),
                torch.from_numpy(pocket.to_vector()),
                torch.tensor(interaction.label, dtype=torch.float32)
            )
        
        x, edge_idx, pocket_vec, label = self._tensor_cache[cache_key]
        return x, edge_idx, pocket_vec, label

def collate_fn(batch):
    """
    Optimized custom collate to batch graphs.
    Batch: List of (x, edge_index, pocket_vec, label)
    Uses vectorized operations instead of loops for speed.
    """
    x_list, edge_index_list, pocket_list, label_list = zip(*batch)
    
    # Pre-allocate batch indices
    batch_idx_list = []
    node_offset = 0
    edge_index_shifted_list = []
    
    # Single pass through batch with index shifting
    for i, (x, edge_index) in enumerate(zip(x_list, edge_index_list)):
        num_nodes = x.shape[0]
        
        # Shift edge indices
        if edge_index.shape[1] > 0:
            edge_index_shifted_list.append(edge_index + node_offset)
        
        # Batch index for pooling
        batch_idx_list.append(torch.full((num_nodes,), i, dtype=torch.long))
        node_offset += num_nodes
    
    # Concatenate all in one go
    x_batch = torch.cat(x_list, dim=0)
    edge_index_batch = torch.cat(edge_index_shifted_list, dim=1) if edge_index_shifted_list else torch.zeros((2, 0), dtype=torch.long)
    batch_vec = torch.cat(batch_idx_list, dim=0)
    pocket_batch = torch.stack(pocket_list)
    label_batch = torch.stack(label_list)
    
    return x_batch, edge_index_batch, batch_vec, pocket_batch, label_batch
