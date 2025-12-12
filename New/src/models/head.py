"""Full end-to-end model."""

import torch
import torch.nn as nn
from src.models.drug_gnn import DrugGNN
from src.models.protein_encoder import ProteinContactGraphEncoder, ProteinCNNEncoder
from src.models.quantum_vqc import QuantumVQC, QuantumKernelClassifier


class DrugProteinInteractionModel(nn.Module):
    """
    Full model combining:
    - Drug GNN encoder
    - Protein encoder (contact graph or CNN)
    - Quantum interaction layer
    - Scoring head
    """
    
    def __init__(
        self,
        config: dict,
        drug_gnn_config: dict = None,
        protein_config: dict = None,
        quantum_config: dict = None,
        head_config: dict = None
    ):
        super().__init__()
        
        # Default configs
        if drug_gnn_config is None:
            drug_gnn_config = config.get('model', {}).get('drug_gnn', {})
        if protein_config is None:
            protein_config = config.get('model', {}).get('protein', {})
        if quantum_config is None:
            quantum_config = config.get('model', {}).get('quantum', {})
        if head_config is None:
            head_config = config.get('model', {}).get('head', {})
        
        dim = config.get('model', {}).get('dim', 256)
        
        # Drug encoder
        self.drug_encoder = DrugGNN(
            in_channels=31,
            edge_in_channels=6,
            hidden_channels=dim,
            out_channels=dim,
            num_layers=drug_gnn_config.get('layers', 4),
            conv_type=drug_gnn_config.get('conv', 'gin'),
            dropout=drug_gnn_config.get('dropout', 0.1)
        )
        
        # Protein encoder
        protein_method = protein_config.get('method', 'contact_gnn')
        if protein_method == 'contact_gnn':
            self.protein_encoder = ProteinContactGraphEncoder(
                in_channels=21,
                hidden_channels=dim,
                out_channels=dim,
                num_layers=3,
                dropout=protein_config.get('dropout', 0.1)
            )
        elif protein_method == 'seq_cnn':
            self.protein_encoder = ProteinCNNEncoder(
                vocab_size=21,
                embedding_dim=128,
                hidden_channels=dim,
                out_channels=dim,
                dropout=protein_config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unknown protein method: {protein_method}")
        
        # Quantum interaction layer
        quantum_mode = quantum_config.get('mode', 'vqc')
        if quantum_mode == 'vqc':
            self.quantum_layer = QuantumVQC(
                input_dim=2 * dim,
                n_qubits=quantum_config.get('n_qubits', 8),
                depth=quantum_config.get('depth', 3),
                backend=quantum_config.get('backend', 'default.qubit'),
                entangler=quantum_config.get('entangler', 'cz'),
                output_dim=1
            )
        elif quantum_mode == 'kernel':
            self.quantum_layer = QuantumKernelClassifier(
                input_dim=2 * dim,
                n_qubits=quantum_config.get('n_qubits', 8),
                backend=quantum_config.get('backend', 'default.qubit'),
                entangler=quantum_config.get('entangler', 'cz')
            )
        else:
            raise ValueError(f"Unknown quantum mode: {quantum_mode}")
        
        self.config = config
        self.drug_gnn_config = drug_gnn_config
        self.protein_config = protein_config
        self.quantum_config = quantum_config
    
    def forward(self, drug_graph, protein_data):
        """
        Args:
            drug_graph: PyG Data object for drug
            protein_data: PyG Data object for protein (or seq tensor if CNN)
        
        Returns:
            scores: (batch_size,)
            quantum_vals: quantum layer output (depends on mode)
            explanations: dict with attention/gradient info
        """
        # Encode drug
        z_drug = self.drug_encoder(drug_graph)  # (B, dim)
        
        # Encode protein
        z_prot = self.protein_encoder(protein_data)  # (B, dim)
        
        # Quantum interaction
        scores, quantum_vals = self.quantum_layer(z_drug, z_prot)
        
        return scores, quantum_vals, {
            'z_drug': z_drug,
            'z_prot': z_prot,
            'quantum_vals': quantum_vals
        }


class ClassicalOnlyModel(nn.Module):
    """Classical baseline (no quantum layer)."""
    
    def __init__(self, config: dict):
        super().__init__()
        
        dim = config.get('model', {}).get('dim', 256)
        
        # Drug encoder
        self.drug_encoder = DrugGNN(
            in_channels=31,
            edge_in_channels=6,
            hidden_channels=dim,
            out_channels=dim,
            num_layers=config['model']['drug_gnn']['layers']
        )
        
        # Protein encoder
        self.protein_encoder = ProteinContactGraphEncoder(
            in_channels=21,
            hidden_channels=dim,
            out_channels=dim
        )
        
        # Score head
        head_config = config['model']['head']
        hidden_dims = head_config.get('hidden_dims', [128, 64])
        dropout = head_config.get('dropout', 0.1)
        
        layers = []
        prev_dim = 2 * dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.score_head = nn.Sequential(*layers)
    
    def forward(self, drug_graph, protein_data):
        """Returns: scores, None, explanations"""
        z_drug = self.drug_encoder(drug_graph)
        z_prot = self.protein_encoder(protein_data)
        
        z_combined = torch.cat([z_drug, z_prot], dim=-1)
        scores = self.score_head(z_combined).squeeze(-1)
        
        return scores, None, {
            'z_drug': z_drug,
            'z_prot': z_prot
        }
