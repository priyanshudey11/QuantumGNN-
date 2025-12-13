"""Quantum VQC layer using PennyLane."""

import torch
import torch.nn as nn
import numpy as np
import warnings

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    warnings.warn("PennyLane not installed; QuantumVQC will be unavailable")


class QuantumVQC(nn.Module):
    """
    Variational Quantum Circuit for interaction scoring.
    
    Input: concatenated [drug_embedding, protein_embedding]
    Output: quantum expectation values -> MLP head -> score
    """
    
    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 8,
        depth: int = 3,
        backend: str = "default.qubit",
        entangler: str = "cz",  # 'cz' or 'cnot'
        output_dim: int = 1
    ):
        super().__init__()
        
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane is required for QuantumVQC")
        
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = backend
        self.entangler = entangler
        
        # Embed input to qubit angles
        self.angle_embed = nn.Linear(input_dim, n_qubits)
        
        # Variational parameters
        self.theta = nn.Parameter(
            torch.randn(depth, n_qubits, 3)  # RY, RZ angles per qubit per layer
        )
        
        # Create quantum device
        self.dev = qml.device(backend, wires=n_qubits)
        
        # Define quantum circuit
        self._build_circuit()
        
        # Output MLP
        self.head = nn.Sequential(
            nn.Linear(n_qubits, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
    
    def _build_circuit(self):
        """Build PennyLane QNode."""
        
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(angles, params):
            # Angle encoding
            for i, angle in enumerate(angles):
                # Normalize angle to [-π, π]
                normalized = torch.tanh(angle) * np.pi
                qml.RY(normalized, wires=i)
            
            # Variational layers
            for layer in range(self.depth):
                # Single-qubit rotations
                for i in range(self.n_qubits):
                    qml.RY(params[layer, i, 0], wires=i)
                    qml.RZ(params[layer, i, 1], wires=i)
                
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    if self.entangler == "cz":
                        qml.CZ(wires=[i, i + 1])
                    elif self.entangler == "cnot":
                        qml.CNOT(wires=[i, i + 1])
                
                # Additional RY at end of layer
                for i in range(self.n_qubits):
                    qml.RY(params[layer, i, 2], wires=i)
            
            # Measurement: return all Pauli-Z expectations
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.circuit = circuit
    
    def forward(self, z_drug, z_prot):
        """
        Args:
            z_drug: (batch_size, dim) drug embedding
            z_prot: (batch_size, dim) protein embedding
        
        Returns:
            scores: (batch_size, 1) interaction scores
            q_vals: (batch_size, n_qubits) quantum expectation values
        """
        # Concatenate embeddings
        z = torch.cat([z_drug, z_prot], dim=-1)  # (B, 2*dim)
        
        # Embed to angles
        angles = self.angle_embed(z)  # (B, n_qubits)
        
        # Run circuit for each sample in batch
        batch_size = angles.shape[0]
        q_vals = []
        
        for i in range(batch_size):
            try:
                # Execute circuit
                expvals = self.circuit(angles[i], self.theta)
                q_vals.append(torch.stack(expvals))
            except Exception as e:
                # Fallback: return zeros if circuit fails
                print(f"Warning: circuit failed for sample {i}: {e}")
                q_vals.append(torch.zeros(self.n_qubits, device=angles.device))
        
        q_vals = torch.stack(q_vals)  # (B, n_qubits)

        # Ensure float32 dtype (PennyLane may return float64)
        q_vals = q_vals.float()

        # Score head
        scores = self.head(q_vals)  # (B, 1)

        return scores.squeeze(-1), q_vals


class QuantumKernelClassifier(nn.Module):
    """
    Quantum Kernel Classifier for interaction scoring.
    
    Uses quantum feature map to compute kernel matrix,
    then trains classical classifier on top.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 8,
        backend: str = "default.qubit",
        entangler: str = "cz"
    ):
        super().__init__()
        
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane is required for QuantumKernelClassifier")
        
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.backend = backend
        
        # Embed input to angles
        self.angle_embed = nn.Linear(input_dim, n_qubits)
        
        # Create quantum device
        self.dev = qml.device(backend, wires=n_qubits)
        
        # Build feature map
        self._build_feature_map(entangler)
        
        # Classical classifier head (on kernel features)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _build_feature_map(self, entangler):
        """Build quantum feature map."""
        
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def feature_map(angles):
            # Angle encoding
            for i, angle in enumerate(angles):
                normalized = torch.tanh(angle) * np.pi
                qml.RY(normalized, wires=i)
            
            # Entangling layer
            for i in range(self.n_qubits - 1):
                if entangler == "cz":
                    qml.CZ(wires=[i, i + 1])
                elif entangler == "cnot":
                    qml.CNOT(wires=[i, i + 1])
            
            # Return probabilities (as proxy for kernel)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.feature_map = feature_map
    
    def forward(self, z_drug, z_prot):
        """
        Args:
            z_drug: (batch_size, dim)
            z_prot: (batch_size, dim)
        
        Returns:
            scores: (batch_size,) interaction scores
        """
        z = torch.cat([z_drug, z_prot], dim=-1)
        angles = self.angle_embed(z)
        
        batch_size = angles.shape[0]
        kernel_features = []
        
        for i in range(batch_size):
            try:
                features = self.feature_map(angles[i])
                kernel_features.append(torch.stack(features))
            except Exception as e:
                print(f"Warning: feature map failed for sample {i}: {e}")
                kernel_features.append(torch.zeros(self.n_qubits, device=angles.device))
        
        kernel_features = torch.stack(kernel_features)
        scores = self.classifier(kernel_features).squeeze(-1)
        
        return scores, kernel_features
