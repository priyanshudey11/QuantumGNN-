import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

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
        
        # Check for MPS device (which doesn't support sparse mm fully)
        is_mps = x.device.type == 'mps'
        
        if is_mps:
            # Fallback to CPU for sparse operations
            cpu_full_edge_index = full_edge_index.cpu()
            cpu_norm = norm.cpu()
            cpu_x = x.cpu()
            
            # Sparse Matrix Multiplication on CPU
            adj = torch.sparse_coo_tensor(
                cpu_full_edge_index, cpu_norm, (num_nodes, num_nodes)
            )
            support = torch.sparse.mm(adj, cpu_x)
            
            # Move back to MPS
            support = support.to(x.device)
        else:
            # Standard execution for CPU/CUDA
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

class QuantumInteractionLayer(nn.Module):
    # Processes multiple samples efficiently using vectorized quantum operations.
    def __init__(self, n_qubits, n_layers, device_name='lightning.qubit'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_name = device_name

        # Don't initialize quantum device/qnode in __init__ - do it lazily in forward
        # This allows the module to be pickled for multiprocessing
        self._dev = None
        self._q_layer = None
        self._initialized = False

        print(f"Quantum layer created (will initialize on first forward pass)")
        print(f"  Qubits: {n_qubits}, Layers: {n_layers}, Device: {device_name}")

    def _lazy_init(self):
        """Initialize quantum circuit on first use (makes it picklable for DataLoader workers)."""
        if self._initialized:
            return

        # Set OpenMP threads for parallel quantum simulation
        import os
        n_threads = os.cpu_count()
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        print(f"  Setting OMP_NUM_THREADS={n_threads} for parallel quantum simulation")

        # Try to use specified device, fallback to lightning.qubit
        try:
            self._dev = qml.device(self.device_name, wires=self.n_qubits)
        except Exception as e:
            fallback = 'lightning.qubit'
            print(f"Warning: {self.device_name} not available ({e}), using {fallback}")
            self._dev = qml.device(fallback, wires=self.n_qubits)

        # Define single circuit for batch processing
        @qml.qnode(self._dev, interface='torch', diff_method='best')
        def circuit(inputs, weights):
            # Encode: Angle Embedding (faster than RX/RY individually)
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))

            # Variational ansatz: Strongly entangling (good expressiveness)
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            # Measure Z on first qubit (single observable = fast)
            return qml.expval(qml.PauliZ(0))

        # Initialize TorchLayer with weights
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        self._q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        self._initialized = True

    def forward(self, x):
        # x shape: (batch_size, n_qubits)
        # Lazy initialization on first forward pass (enables OpenMP parallelization)
        self._lazy_init()

        # TorchLayer handles batching; lightning.qubit uses OpenMP for parallel execution
        outputs = self._q_layer(x)  # Shape: (batch_size,)
        return outputs.unsqueeze(1) if outputs.dim() == 1 else outputs

    def state_dict(self, *args, **kwargs):
        # Override state_dict to handle lazy initialization.
        if not self._initialized:
            self._lazy_init()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # Override load_state_dict to handle lazy initialization.
        if not self._initialized:
            self._lazy_init()
        return super().load_state_dict(state_dict, *args, **kwargs)

class LigandPocketQGNN(nn.Module):
    # Composite QGNN Model.
    def __init__(self,
                 ligand_in_dim,
                 pocket_in_dim,
                 hidden_dim=64,
                 latent_dim=6,
                 n_qubits=6,
                 n_qlayers=2,
                 use_quantum=True,
                 quantum_device='lightning.qubit',
                 use_parallel=True): 
        super().__init__()

        self.use_quantum = use_quantum
        self.latent_dim = latent_dim
        self.half_dim = n_qubits // 2

        self.ligand_encoder = LigandGNN(ligand_in_dim, hidden_dim, self.half_dim)
        self.pocket_encoder = PocketMLP(pocket_in_dim, hidden_dim, self.half_dim)

        if use_quantum:
            if use_parallel:
                # Use parallel quantum layer for multi-core processing
                from .quantum_parallel import ParallelQuantumInteractionLayer
                self.interaction = ParallelQuantumInteractionLayer(
                    n_qubits, n_qlayers, device_name=quantum_device
                )
            else:
                # Use standard quantum layer (single-threaded)
                self.interaction = QuantumInteractionLayer(
                    n_qubits, n_qlayers, device_name=quantum_device
                )
        else:
            self.interaction = nn.Sequential(
                nn.Linear(n_qubits, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            
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
