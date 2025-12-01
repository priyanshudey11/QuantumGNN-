import torch
import pennylane as qml
import numpy as np

# Check for GPU
try:
    import pennylane as qml
    if 'lightning.gpu' in qml.plugin_devices:
        device_name = 'lightning.gpu'
        print("✓ Using lightning.gpu")
    else:
        print("⚠ lightning.gpu not found, skipping test")
        exit(0)
except ImportError:
    print("PennyLane not found")
    exit(0)

# Parameters
num_qubits = 6
total_wires = 2 * num_qubits
batch_size = 128
ligand_dim = 11
pocket_dim = 19

# Dummy Data (on GPU)
device = torch.device('cuda')
ligand_features = torch.randn(batch_size, num_qubits, device=device)
pocket_features = torch.randn(batch_size, num_qubits, device=device)
q_params = torch.randn(2 * total_wires * 2, device=device) # 2 layers

# Define Circuit
dev = qml.device(device_name, wires=total_wires)

@qml.qnode(dev, interface='torch')
def circuit(ligand_f, pocket_f, params):
    # Reshape params
    params = params.reshape(2, total_wires, 2)
    
    # 1. Angle encoding (Vectorized)
    for i in range(num_qubits):
        qml.RY(ligand_f[:, i], wires=i)
        
    for i in range(num_qubits):
        qml.RY(pocket_f[:, i], wires=num_qubits + i)
        
    # 2. Entanglement
    for i in range(num_qubits):
        qml.CNOT(wires=[i, num_qubits + i])
        
    # 3. Variational
    for layer in range(2):
        for wire in range(total_wires):
            qml.RZ(params[layer, wire, 0], wires=wire)
            qml.RX(params[layer, wire, 1], wires=wire)
            
    return [qml.expval(qml.PauliZ(i)) for i in range(total_wires)]

print("Running Circuit...")
try:
    results = circuit(ligand_features, pocket_features, q_params)
    # Stack results
    if isinstance(results, (list, tuple)):
        results = torch.stack(results, dim=1)
    
    print(f"✓ Circuit Success! Output shape: {results.shape}")
    print(f"Output device: {results.device}")
    
except Exception as e:
    print(f"❌ Circuit Failed: {e}")
    import traceback
    traceback.print_exc()
