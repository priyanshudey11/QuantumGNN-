"""
Quantum Graph Neural Network model for drug-patient interaction prediction.

This module implements:
- QuantumDrugPatientGNN: Main model class
- Quantum interaction layer using PennyLane
- Classical fallback mode for comparison

Mathematical formulation:

1. Classical pre-embedding:
   h_D = σ(W_D * x_D + b_D)
   h_P = σ(W_P * x_P + b_P)

2. Quantum encoding (angle encoding):
   |ψ_D(0)⟩ = ⊗_k R_y(h_D[k])|0⟩
   |ψ_P(0)⟩ = ⊗_k R_y(h_P[k])|0⟩

3. Entangling ansatz:
   U_DP(Θ) = ∏_wires (R_z(φ) R_x(λ)) · ∏_pairs CNOT

4. Readout:
   z = ⟨ψ_DP_final | Z | ψ_DP_final⟩

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
    """Quantum circuit layer for encoding drug-patient interactions.

    Uses PennyLane to create a variational quantum circuit that:
    1. Encodes drug features on first half of qubits
    2. Encodes patient features on second half of qubits
    3. Applies entangling gates between drug and patient qubits
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

    def _quantum_circuit(self, drug_features, patient_features, q_params):
        """Quantum circuit implementation.

        Circuit structure:
        1. Angle encoding: R_y rotations for features
        2. Entangling layer: CNOTs between drug and patient qubits
        3. Variational layers: R_z, R_x rotations with learned parameters
        4. Measurement: PauliZ expectations

        # ═══════════════════════════════════════════════════════════════════
        # QUBIT MAPPING (for a single drug-patient interaction edge):
        # ═══════════════════════════════════════════════════════════════════
        #
        # Graph features → Classical encoders → Latent vectors → Quantum wires
        #
        # Drug node features (drug_dim=19):
        #   → MLP encoder → h_drug[0..num_qubits-1] (latent vector)
        #   → Angle encoding: RY(h_drug[i]) on wire i
        #   → Mapped to wires [0, 1, ..., num_qubits-1]
        #
        # Patient node features (patient_dim=41):
        #   → MLP encoder → h_patient[0..num_qubits-1] (latent vector)
        #   → Angle encoding: RY(h_patient[i]) on wire (num_qubits + i)
        #   → Mapped to wires [num_qubits, num_qubits+1, ..., 2*num_qubits-1]
        #
        # Example with num_qubits=4 (total 8 wires):
        #   wires[0..3]  = drug latent features h_drug[0], h_drug[1], h_drug[2], h_drug[3]
        #   wires[4..7]  = patient latent features h_patient[0], h_patient[1], h_patient[2], h_patient[3]
        #
        # Entanglement:
        #   - Initial: CNOT(0→4), CNOT(1→5), CNOT(2→6), CNOT(3→7)
        #              (pairs each drug wire with its corresponding patient wire)
        #   - Variational: Ring topology CNOT(i→i+1) for all wires
        #
        # ⚠️ Important: This is NOT a 1-to-1 vertex-to-qubit mapping!
        #    - Each drug-patient PAIR occupies 2*num_qubits wires
        #    - High-dimensional node features are compressed to num_qubits latent dims
        #    - Each latent dimension is encoded on ONE wire via angle encoding
        # ═══════════════════════════════════════════════════════════════════

        Args:
            drug_features: Tensor of shape (num_qubits,) - drug latent vector
            patient_features: Tensor of shape (num_qubits,) - patient latent vector
            q_params: Variational parameters

        Returns:
            List of PauliZ expectation values
        """
        # Reshape parameters
        params = q_params.reshape(self.num_qlayers, self.total_wires, 2)

        # 1. Angle encoding for drug features (first half of qubits)
        for i in range(self.num_qubits):
            qml.RY(drug_features[i], wires=i)

        # 2. Angle encoding for patient features (second half of qubits)
        for i in range(self.num_qubits):
            qml.RY(patient_features[i], wires=self.num_qubits + i)

        # 3. Initial entangling layer (connect drug and patient qubits)
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

    def forward(self, drug_features, patient_features):
        """Forward pass through quantum circuit.

        Args:
            drug_features: Tensor of shape (batch_size, num_qubits)
            patient_features: Tensor of shape (batch_size, num_qubits)

        Returns:
            Tensor of shape (batch_size, total_wires) with quantum measurements
        """
        batch_size = drug_features.shape[0]
        outputs = []

        # Process each sample in the batch
        for i in range(batch_size):
            measurements = self.qnode(
                drug_features[i],
                patient_features[i],
                self.q_params
            )
            outputs.append(torch.stack(measurements))
        stacked = torch.stack(outputs)

        # Match the dtype/device of upstream tensors to avoid precision mismatches downstream.
        return stacked.to(dtype=drug_features.dtype, device=drug_features.device)


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

    def forward(self, drug_features, patient_features):
        """Forward pass through classical network.

        Args:
            drug_features: Tensor of shape (batch_size, num_qubits)
            patient_features: Tensor of shape (batch_size, num_qubits)

        Returns:
            Tensor of shape (batch_size, 2 * num_qubits)
        """
        # Concatenate drug and patient features
        combined = torch.cat([drug_features, patient_features], dim=1)
        return self.network(combined)


class QuantumDrugPatientGNN(nn.Module):
    """Quantum Graph Neural Network for drug-patient interaction prediction.

    Architecture:
    1. Drug encoder: projects drug features to latent space
    2. Patient encoder: projects patient features to latent space
    3. Quantum/Classical interaction layer: models drug-patient interactions
    4. Output head: predicts outcome probability

    Args:
        drug_dim: Dimension of drug input features
        patient_dim: Dimension of patient input features
        num_qubits: Number of qubits per side (drug + patient)
        num_qlayers: Number of variational quantum layers
        hidden_dim: Hidden dimension for encoders (default: 64)
        use_quantum: Whether to use quantum circuit (True) or classical fallback (False)
        device_name: PennyLane device name for quantum mode
    """

    def __init__(
        self,
        drug_dim: int,
        patient_dim: int,
        num_qubits: int = 4,
        num_qlayers: int = 2,
        hidden_dim: int = 64,
        use_quantum: bool = True,
        device_name: str = 'default.qubit'
    ):
        super().__init__()

        self.drug_dim = drug_dim
        self.patient_dim = patient_dim
        self.num_qubits = num_qubits
        self.num_qlayers = num_qlayers
        self.use_quantum = use_quantum

        # Drug encoder: drug_dim → hidden_dim → num_qubits
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_qubits)
        )

        # Patient encoder: patient_dim → hidden_dim → num_qubits
        self.patient_encoder = nn.Sequential(
            nn.Linear(patient_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_qubits)
        )

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

        # Output head: interaction_output → 32 → 1 (probability)
        self.output_head = nn.Sequential(
            nn.Linear(interaction_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, drug_features, patient_features):
        """Forward pass through the QGNN.

        Args:
            drug_features: Tensor of shape (batch_size, drug_dim)
            patient_features: Tensor of shape (batch_size, patient_dim)

        Returns:
            Tensor of shape (batch_size, 1) with outcome probabilities
        """
        # Encode features to latent space
        h_drug = self.drug_encoder(drug_features)  # (batch_size, num_qubits)
        h_patient = self.patient_encoder(patient_features)  # (batch_size, num_qubits)

        # Apply tanh to map to [-π, π] for quantum encoding
        if self.use_quantum:
            h_drug = torch.tanh(h_drug) * np.pi
            h_patient = torch.tanh(h_patient) * np.pi

        # Quantum/classical interaction layer
        interaction_output = self.interaction_layer(h_drug, h_patient)

        # Output prediction
        output = self.output_head(interaction_output)

        return output

    def predict_batch(self, drug_features, patient_features, threshold: float = 0.5):
        """Predict outcomes for a batch of drug-patient pairs.

        Args:
            drug_features: Tensor of shape (batch_size, drug_dim)
            patient_features: Tensor of shape (batch_size, patient_dim)
            threshold: Classification threshold (default: 0.5)

        Returns:
            Tuple of (probabilities, predictions)
            - probabilities: Tensor of shape (batch_size,)
            - predictions: Tensor of shape (batch_size,) with binary predictions
        """
        self.eval()
        with torch.no_grad():
            probs = self(drug_features, patient_features).squeeze(-1)
            preds = (probs >= threshold).long()
        return probs, preds

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> dict:
        """Get model configuration information."""
        return {
            'drug_dim': self.drug_dim,
            'patient_dim': self.patient_dim,
            'num_qubits': self.num_qubits,
            'num_qlayers': self.num_qlayers,
            'use_quantum': self.use_quantum,
            'num_parameters': self.get_num_parameters()
        }
