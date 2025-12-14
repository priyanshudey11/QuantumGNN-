import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session
from qiskit_ibm_runtime import Options

class GCNLayer(nn.Module):
    # Simple GCN Layer: H' = ReLU(D^-0.5 A D^-0.5 H W).
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        # x: (N, in_features)
        # edge_index: (2, E)

        num_nodes = x.shape[0]

        # Create adjacency matrix (sparse)
        # Add self-loops
        loops = torch.arange(num_nodes, device=x.device)
        loop_index = torch.stack([loops, loops])

        if edge_index.shape[1] > 0:
            full_edge_index = torch.cat([edge_index, loop_index], dim=1)
        else:
            full_edge_index = loop_index

        # Compute normalization
        row, col = full_edge_index
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones(full_edge_index.shape[1], device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Message passing: norm[row] * norm[col]
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Sparse Matrix Multiplication: A * X
        # PyTorch sparse tensors are (indices, values, size)
        adj = torch.sparse_coo_tensor(
            full_edge_index, norm, (num_nodes, num_nodes)
        )

        # Support = A * X
        support = torch.sparse.mm(adj, x)

        # Output = Support * W
        out = self.linear(support)
        return out

class LigandGNN(nn.Module):
    # Classical GNN to encode Ligand Graph into a vector.
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNLayer(input_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # x: Node features
        # edge_index: Graph connectivity
        # batch: Batch vector mapping nodes to graphs

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Global mean pooling
        # Scatter add features by batch index
        batch_size = batch.max().item() + 1
        pooled = torch.zeros(batch_size, x.shape[1], device=x.device)
        pooled.scatter_add_(0, batch.unsqueeze(1).expand(-1, x.shape[1]), x)

        # Divide by counts
        counts = torch.zeros(batch_size, device=x.device)
        counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        pooled = pooled / counts.unsqueeze(1).clamp(min=1)

        # Final projection
        x = self.lin(pooled)
        return x

class PocketMLP(nn.Module):
    # Classical MLP to encode Pocket Features into a vector.
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class IBMQuantumInteractionLayer(nn.Module):
    # Quantum Circuit to model interaction using IBM Quantum hardware.
    def __init__(self, n_qubits, n_layers, ibm_backend='ibmq_qasm_simulator', use_session=True, instance=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ibm_backend = ibm_backend
        self.use_session = use_session

        # Initialize IBM Quantum Runtime Service
        print(f"Initializing IBM Quantum backend: {ibm_backend}")
        try:
            # If instance is provided directly, use it to avoid API timeout issues
            if instance is not None:
                print(f"  Using provided instance: {instance}")
                self.service = QiskitRuntimeService(instance=instance)
            else:
                # Initialize service and get available instances
                self.service = QiskitRuntimeService()

                # Get available instances and use the first one (or set a default)
                try:
                    instances = self.service.instances()
                    if instances:
                        instance_crn = instances[0]['crn']
                        print(f"  Using instance: {instances[0]['name']} ({instances[0]['plan']})")
                        # Reinitialize with the specific instance
                        self.service = QiskitRuntimeService(instance=instance_crn)
                    else:
                        print(f"  Warning: No instances found, using default configuration")
                except Exception as inst_error:
                    print(f"  Warning: Could not retrieve instances ({inst_error})")
                    print(f"  Continuing with default service configuration...")

            self.backend = self.service.backend(ibm_backend)
            print(f" Connected to IBM backend: {self.backend.name}")
            print(f"  Status: {self.backend.status().status_msg}")
            print(f"  Qubits: {self.backend.num_qubits}")
            print(f"  Pending jobs: {self.backend.status().pending_jobs}")
        except Exception as e:
            print(f"⚠ Error connecting to IBM backend: {e}")
            print(f"  Please ensure your account is saved correctly")
            print(f"  Run in notebook: QiskitRuntimeService.save_account(token='YOUR_IBM_TOKEN', overwrite=True)")
            raise

        # Create PennyLane device using IBM backend object
        self.dev = qml.device(
            'qiskit.remote',
            wires=n_qubits,
            backend=self.backend,  # Pass the backend object, not the string name
            ibmqx_token=None,  # Uses saved credentials
            shots=8192  # Number of shots for measurement
        )

        print(f" PennyLane device initialized with IBM backend")

        # Define Quantum Node
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def circuit(inputs, weights):
            # Encoding (Angle Embedding)
            qml.AngleEmbedding(inputs, wires=range(n_qubits))

            # Variational Layers (StronglyEntanglingLayers ansatz)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            # Measurement
            return qml.expval(qml.PauliZ(0))

        self.qnode = circuit

        # Weight shape for StronglyEntanglingLayers
        # Each layer: n_qubits × 3 (RZ-RY-RZ rotations)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

        print(f"  Circuit depth: {n_layers} layers")
        print(f"  Trainable parameters: {n_layers * n_qubits * 3}")
        print(f"  Ansatz: StronglyEntanglingLayers")
        print(f"  Differentiation: parameter-shift rule")
        print(f"  Shots per measurement: 8192")

    def forward(self, x):
        return self.q_layer(x)

class LigandPocketQGNN_IBM(nn.Module):
    """Composite QGNN Model using IBM Quantum Hardware."""
    def __init__(self,
                 ligand_in_dim,
                 pocket_in_dim,
                 hidden_dim=64,
                 latent_dim=6,
                 n_qubits=6,
                 n_qlayers=2,
                 use_quantum=True,
                 ibm_backend='ibmq_qasm_simulator',
                 use_session=True,
                 instance=None):
        super().__init__()

        self.use_quantum = use_quantum
        self.latent_dim = latent_dim
        self.half_dim = n_qubits // 2

        self.ligand_encoder = LigandGNN(ligand_in_dim, hidden_dim, self.half_dim)
        self.pocket_encoder = PocketMLP(pocket_in_dim, hidden_dim, self.half_dim)

        if use_quantum:
                       print("IBM QUANTUM BACKEND CONFIGURATION")
                        self.interaction = IBMQuantumInteractionLayer(
                n_qubits,
                n_qlayers,
                ibm_backend=ibm_backend,
                use_session=use_session,
                instance=instance
            )
                    else:
            # Classical fallback
            self.interaction = nn.Sequential(
                nn.Linear(n_qubits, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            print(" Using classical MLP for interaction layer")

    def forward(self, x, edge_index, batch, pocket_vec):
        # Encode Ligand
        h_ligand = self.ligand_encoder(x, edge_index, batch)

        # Encode Pocket
        h_pocket = self.pocket_encoder(pocket_vec)

        # Combine
        combined = torch.cat([h_ligand, h_pocket], dim=1)

        # Normalize for Angle Embedding (0 to 2pi)
        combined = torch.tanh(combined) * torch.pi

        # Interaction
        out = self.interaction(combined)

        if self.use_quantum:
            # Map expectation [-1, 1] to [0, 1]
            out = (out + 1) / 2

        return out
