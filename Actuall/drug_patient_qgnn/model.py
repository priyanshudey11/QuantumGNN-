"""
Quantum Graph Neural Network model for drug-protein interaction prediction.

This module implements:
- QuantumDrugProteinGNN: Main model class
- Quantum interaction layer using PennyLane
- Classical fallback mode for comparison

Mathematical formulation:

1. Classical pre-embedding:
   h_L = σ(W_L * x_L + b_L)  # Ligand (drug) features
   h_P = σ(W_P * x_P + b_P)  # Protein pocket features

2. Quantum encoding (angle encoding):
   |ψ_L(0)⟩ = ⊗_k R_y(h_L[k])|0⟩  # Ligand qubits
   |ψ_P(0)⟩ = ⊗_k R_y(h_P[k])|0⟩  # Pocket qubits

3. Entangling ansatz:
   U_LP(Θ) = ∏_wires (R_z(φ) R_x(λ)) · ∏_pairs CNOT

4. Readout:
   z = ⟨ψ_LP_final | Z | ψ_LP_final⟩

5. Classical output head:
   prob = σ(W_out * ReLU(W_1 * z + b_1) + b_out)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("Warning: PennyLane not available. Only classical mode will work.")


class QuantumInteractionLayer(nn.Module):
    """Quantum circuit layer for encoding ligand-protein interactions.

    Uses PennyLane to create a variational quantum circuit that:
    1. Encodes ligand (drug) features on first half of qubits
    2. Encodes protein pocket features on second half of qubits
    3. Applies entangling gates between ligand and pocket qubits
    4. Applies variational layers
    5. Measures PauliZ expectations

    Args:
        num_qubits: Number of qubits per side (total = 2 * num_qubits)
        num_qlayers: Number of variational layers
        device_name: PennyLane device name (default: 'default.qubit')
    """

    def __init__(
        self,
        num_qubits: int = 4,
        num_qlayers: int = 2,
        device_name: str = 'default.qubit'
    ):
        super().__init__()

        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is required for quantum mode. Install with: pip install pennylane")

        self.num_qubits = num_qubits
        self.num_qlayers = num_qlayers
        self.total_wires = 2 * num_qubits

        # Create quantum device
        self.dev = qml.device(device_name, wires=self.total_wires)

        # Variational parameters (num_qlayers × total_wires × 2)
        # Each wire gets (R_z, R_x) parameters per layer
        num_params = num_qlayers * self.total_wires * 2
        self.q_params = nn.Parameter(torch.randn(num_params) * 0.1)

        # Create the quantum circuit
        self.qnode = qml.QNode(self._quantum_circuit, self.dev, interface='torch')

    def _quantum_circuit(self, ligand_features, pocket_features, q_params):
        """Quantum circuit implementation.

        Circuit structure:
        1. Angle encoding: R_y rotations for features
        2. Entangling layer: CNOTs between ligand and pocket qubits
        3. Variational layers: R_z, R_x rotations with learned parameters
        4. Measurement: PauliZ expectations

        # ═══════════════════════════════════════════════════════════════════
        # QUBIT MAPPING (for a single ligand-pocket interaction edge):
        # ═══════════════════════════════════════════════════════════════════
        #
        # Graph features → Classical encoders → Latent vectors → Quantum wires
        #
        # Ligand node features (ligand_dim):
        #   → MLP encoder → h_ligand[0..num_qubits-1] (latent vector)
        #   → Angle encoding: RY(h_ligand[i]) on wire i
        #   → Mapped to wires [0, 1, ..., num_qubits-1]
        #
        # Protein pocket node features (pocket_dim):
        #   → MLP encoder → h_pocket[0..num_qubits-1] (latent vector)
        #   → Angle encoding: RY(h_pocket[i]) on wire (num_qubits + i)
        #   → Mapped to wires [num_qubits, num_qubits+1, ..., 2*num_qubits-1]
        #
        # Example with num_qubits=4 (total 8 wires):
        #   wires[0..3]  = ligand latent features h_ligand[0], h_ligand[1], h_ligand[2], h_ligand[3]
        #   wires[4..7]  = pocket latent features h_pocket[0], h_pocket[1], h_pocket[2], h_pocket[3]
        #
        # Entanglement:
        #   - Initial: CNOT(0→4), CNOT(1→5), CNOT(2→6), CNOT(3→7)
        #              (pairs each ligand wire with its corresponding pocket wire)
        #   - Variational: Ring topology CNOT(i→i+1) for all wires
        #
        # ⚠️ Important: This is NOT a 1-to-1 vertex-to-qubit mapping!
        #    - Each ligand-pocket PAIR occupies 2*num_qubits wires
        #    - High-dimensional node features are compressed to num_qubits latent dims
        #    - Each latent dimension is encoded on ONE wire via angle encoding
        # ═══════════════════════════════════════════════════════════════════

        Args:
            ligand_features: Tensor of shape (num_qubits,) - ligand latent vector
            pocket_features: Tensor of shape (num_qubits,) - pocket latent vector
            q_params: Variational parameters

        Returns:
            List of PauliZ expectation values
        """
        # Reshape parameters
        params = q_params.reshape(self.num_qlayers, self.total_wires, 2)

        # 1. Angle encoding for ligand features (first half of qubits)
        for i in range(self.num_qubits):
            qml.RY(ligand_features[:, i], wires=i)

        # 2. Angle encoding for pocket features (second half of qubits)
        for i in range(self.num_qubits):
            qml.RY(pocket_features[:, i], wires=self.num_qubits + i)

        # 3. Initial entangling layer (connect ligand and pocket qubits)
        for i in range(self.num_qubits):
            qml.CNOT(wires=[i, self.num_qubits + i])

        # 4. Variational layers
        for layer in range(self.num_qlayers):
            # Rotation gates
            for wire in range(self.total_wires):
                qml.RZ(params[layer, wire, 0], wires=wire)
                qml.RX(params[layer, wire, 1], wires=wire)

            # Entangling CNOTs in ring topology
            for wire in range(self.total_wires - 1):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CNOT(wires=[self.total_wires - 1, 0])  # Close the ring

        # 5. Measurements: measure PauliZ on all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.total_wires)]

    def forward(self, ligand_features, pocket_features):
        """Forward pass through quantum circuit.

        Args:
            ligand_features: Tensor of shape (batch_size, num_qubits)
            pocket_features: Tensor of shape (batch_size, num_qubits)

        Returns:
            Tensor of shape (batch_size, total_wires) with quantum measurements
        """
        # Pass the entire batch to the QNode
        # PennyLane handles batching automatically when using torch interface
        measurements = self.qnode(
            ligand_features,
            pocket_features,
            self.q_params
        )
        
        # Stack measurements if returned as a list of tensors (one per wire)
        if isinstance(measurements, (list, tuple)):
            # measurements is [Tensor(batch_size), Tensor(batch_size), ...]
            # Stack to get (batch_size, total_wires)
            stacked = torch.stack(measurements, dim=1)
        else:
            stacked = measurements

        # Match the dtype/device of upstream tensors to avoid precision mismatches downstream.
        return stacked.to(dtype=ligand_features.dtype, device=ligand_features.device)


class ClassicalInteractionLayer(nn.Module):
    """Classical fallback for quantum interaction layer.

    Implements a standard MLP that mimics the quantum layer's interface
    but uses classical neural network operations.

    Args:
        num_qubits: Input dimension per side (total input = 2 * num_qubits)
        num_qlayers: Number of hidden layers
    """

    def __init__(self, num_qubits: int = 4, num_qlayers: int = 2):
        super().__init__()

        self.num_qubits = num_qubits
        input_dim = 2 * num_qubits
        hidden_dim = 2 * num_qubits

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_qlayers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, ligand_features, pocket_features):
        """Forward pass through classical network.

        Args:
            ligand_features: Tensor of shape (batch_size, num_qubits)
            pocket_features: Tensor of shape (batch_size, num_qubits)

        Returns:
            Tensor of shape (batch_size, 2 * num_qubits)
        """
        # Concatenate ligand and pocket features
        combined = torch.cat([ligand_features, pocket_features], dim=1)
        return self.network(combined)


class QuantumDrugProteinGNN(nn.Module):
    """Quantum Graph Neural Network for drug-protein interaction prediction.

    Architecture:
    1. Ligand encoder: projects ligand (drug) features to latent space
    2. Pocket encoder: projects protein pocket features to latent space
    3. Quantum/Classical interaction layer: models ligand-pocket interactions
    4. Output head: predicts binding probability

    Args:
        ligand_dim: Dimension of ligand input features
        pocket_dim: Dimension of protein pocket input features
        num_qubits: Number of qubits per side (ligand + pocket)
        num_qlayers: Number of variational quantum layers
        hidden_dim: Hidden dimension for encoders (default: 64)
        use_quantum: Whether to use quantum circuit (True) or classical fallback (False)
        device_name: PennyLane device name for quantum mode

        # Backward compatibility aliases
        drug_dim: Alias for ligand_dim
        patient_dim: Alias for pocket_dim
    """

    def __init__(
        self,
        ligand_dim: Optional[int] = None,
        pocket_dim: Optional[int] = None,
        num_qubits: int = 4,
        num_qlayers: int = 2,
        hidden_dim: int = 64,
        use_quantum: bool = True,
        device_name: str = 'default.qubit',
        # Backward compatibility
        drug_dim: Optional[int] = None,
        patient_dim: Optional[int] = None
    ):
        super().__init__()

        # Handle backward compatibility
        if drug_dim is not None and ligand_dim is None:
            ligand_dim = drug_dim
        if patient_dim is not None and pocket_dim is None:
            pocket_dim = patient_dim

        if ligand_dim is None or pocket_dim is None:
            raise ValueError("Must provide either (ligand_dim, pocket_dim) or (drug_dim, patient_dim)")

        self.ligand_dim = ligand_dim
        self.pocket_dim = pocket_dim
        # Backward compatibility
        self.drug_dim = ligand_dim
        self.patient_dim = pocket_dim

        self.num_qubits = num_qubits
        self.num_qlayers = num_qlayers
        self.use_quantum = use_quantum

        # Ligand encoder: ligand_dim → hidden_dim → num_qubits
        self.ligand_norm = nn.BatchNorm1d(ligand_dim)
        self.ligand_encoder = nn.Sequential(
            nn.Linear(ligand_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_qubits)
        )

        # Pocket encoder: pocket_dim → hidden_dim → num_qubits
        self.pocket_norm = nn.BatchNorm1d(pocket_dim)
        self.pocket_encoder = nn.Sequential(
            nn.Linear(pocket_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_qubits)
        )

        # Backward compatibility aliases
        self.drug_encoder = self.ligand_encoder
        self.patient_encoder = self.pocket_encoder

        # Quantum or classical interaction layer
        if use_quantum:
            self.interaction_layer = QuantumInteractionLayer(
                num_qubits=num_qubits,
                num_qlayers=num_qlayers,
                device_name=device_name
            )
            interaction_output_dim = 2 * num_qubits
        else:
            self.interaction_layer = ClassicalInteractionLayer(
                num_qubits=num_qubits,
                num_qlayers=num_qlayers
            )
            interaction_output_dim = 2 * num_qubits

        # Output head: interaction_output → 32 → 1 (logits)
        self.output_head = nn.Sequential(
            nn.Linear(interaction_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, ligand_features, pocket_features):
        """Forward pass through the QGNN.

        Args:
            ligand_features: Tensor of shape (batch_size, ligand_dim)
            pocket_features: Tensor of shape (batch_size, pocket_dim)

        Returns:
            Tensor of shape (batch_size, 1) with binding probabilities
        """
        # Encode features to latent space
        # Apply normalization first
        ligand_features = self.ligand_norm(ligand_features)
        pocket_features = self.pocket_norm(pocket_features)
        
        h_ligand = self.ligand_encoder(ligand_features)  # (batch_size, num_qubits)
        h_pocket = self.pocket_encoder(pocket_features)  # (batch_size, num_qubits)

        # Apply tanh to map to [-π, π] for quantum encoding
        if self.use_quantum:
            h_ligand = torch.tanh(h_ligand) * np.pi
            h_pocket = torch.tanh(h_pocket) * np.pi

        # Quantum/classical interaction layer
        interaction_output = self.interaction_layer(h_ligand, h_pocket)

        # Output prediction
        output = self.output_head(interaction_output)

        return output

    def predict_batch(self, ligand_features, pocket_features, threshold: float = 0.5):
        """Predict binding for a batch of ligand-pocket pairs.

        Args:
            ligand_features: Tensor of shape (batch_size, ligand_dim)
            pocket_features: Tensor of shape (batch_size, pocket_dim)
            threshold: Classification threshold (default: 0.5)

        Returns:
            Tuple of (probabilities, predictions)
            - probabilities: Tensor of shape (batch_size,)
            - predictions: Tensor of shape (batch_size,) with binary predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self(ligand_features, pocket_features).squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
        return probs, preds

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> dict:
        """Get model configuration information."""
        return {
            'ligand_dim': self.ligand_dim,
            'pocket_dim': self.pocket_dim,
            'num_qubits': self.num_qubits,
            'num_qlayers': self.num_qlayers,
            'use_quantum': self.use_quantum,
            'num_parameters': self.get_num_parameters(),
            # Backward compatibility
            'drug_dim': self.drug_dim,
            'patient_dim': self.patient_dim
        }


# Backward compatibility alias
QuantumDrugPatientGNN = QuantumDrugProteinGNN
