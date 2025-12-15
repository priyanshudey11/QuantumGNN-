import torch
import torch.nn as nn
import pennylane as qml
from concurrent.futures import ThreadPoolExecutor
import os


class ParallelQuantumInteractionLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, device_name='default.qubit'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_name = device_name

        # Lazy initialization for pickling support
        self._dev = None
        self._qnode = None
        self._weights = None
        self._initialized = False

        # Thread pool for parallel evaluation
        n_workers = os.cpu_count()
        self._executor = ThreadPoolExecutor(max_workers=n_workers)

        print(f"  Parallel Quantum layer created")
        print(f"  Qubits: {n_qubits}, Layers: {n_layers}, Device: {device_name}")
        print(f"  Using {n_workers} worker threads for parallel evaluation")

    def _lazy_init(self):
        # Initialize quantum circuit on first use.
        if self._initialized:
            return

        # Set environment for maximum parallelism
        n_threads = os.cpu_count()
        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        os.environ['MKL_NUM_THREADS'] = str(n_threads)

        # Create quantum device
        try:
            self._dev = qml.device(self.device_name, wires=self.n_qubits)
        except Exception as e:
            fallback = 'default.qubit'
            print(f"⚠ Warning: {self.device_name} not available, using {fallback}")
            self._dev = qml.device(fallback, wires=self.n_qubits)

        # Define quantum circuit
        @qml.qnode(self._dev, interface='torch', diff_method='best')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))

        self._qnode = circuit

        # Initialize trainable weights as nn.Parameter
        self._weights = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits, 3) * 0.1
        )

        self._initialized = True
        print(f" Quantum circuit initialized with {n_threads} OpenMP threads")

    def forward(self, x):

        # Forward pass with parallel batch evaluation.
        self._lazy_init()

        batch_size = x.shape[0]

        # For tiny batches, sequential is faster (no threading overhead)
        if batch_size <= 2:
            results = [self._qnode(x[i], self._weights) for i in range(batch_size)]
            outputs = torch.stack(results)
            return outputs.unsqueeze(1) if outputs.dim() == 1 else outputs

        # For larger batches, use parallel evaluation across threads
        
        # Move to CPU for threading stability (avoids MPS/CUDA context issues in threads)
        x_cpu = x.cpu()
        weights_cpu = self._weights.cpu()
        
        def eval_single(i):
            return self._qnode(x_cpu[i], weights_cpu)

        # Submit all tasks to thread pool
        futures = [self._executor.submit(eval_single, i) for i in range(batch_size)]

        # Gather results
        results = [future.result() for future in futures]
        outputs = torch.stack(results)
        
        # Move back to original device
        # MPS doesn't support float64, so ensure we cast to match input dtype (usually float32)
        if x.dtype == torch.float32:
            outputs = outputs.to(dtype=torch.float32)
            
        outputs = outputs.to(x.device)

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

    def __del__(self):
        # Clean up thread pool on deletion.
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
