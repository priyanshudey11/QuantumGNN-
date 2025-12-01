#!/usr/bin/env python3
"""
Quick test to verify GPU setup for quantum training.
"""

import torch
import pennylane as qml
from drug_patient_qgnn import QuantumDrugPatientGNN, print_device_info

print("="*70)
print("GPU SETUP TEST")
print("="*70)

# 1. Check PyTorch GPU
print("\n1. PyTorch GPU Status:")
print_device_info()

# 2. Check PennyLane GPU device
print("\n2. PennyLane GPU Device Test:")
try:
    dev = qml.device('lightning.gpu', wires=6)
    print(f"✅ lightning.gpu device created successfully")
    print(f"   Device: {dev}")
    print(f"   Wires: {dev.wires}")
except Exception as e:
    print(f"❌ Error creating lightning.gpu device: {e}")
    print("   Falling back to lightning.qubit (CPU)")
    dev = qml.device('lightning.qubit', wires=6)

# 3. Test quantum circuit on GPU
print("\n3. Testing Quantum Circuit:")
@qml.qnode(dev, interface='torch')
def test_circuit(x):
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

test_input = torch.tensor([0.5, 0.3], device='cuda')
print(f"   Input: {test_input}")
print(f"   Input device: {test_input.device}")

try:
    result = test_circuit(test_input)
    print(f"✅ Circuit executed successfully")
    print(f"   Output: {result}")
    print(f"   Output device: {result.device}")
except Exception as e:
    print(f"❌ Circuit execution failed: {e}")

# 4. Test model creation
print("\n4. Testing Model Creation:")
try:
    model = QuantumDrugPatientGNN(
        drug_dim=11,
        patient_dim=19,
        num_qubits=6,
        num_qlayers=2,
        hidden_dim=128,
        use_quantum=True,
        device_name='lightning.gpu'
    )
    model = model.to('cuda')
    print(f"✅ Quantum model created successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    drug_batch = torch.randn(8, 11, device='cuda')
    patient_batch = torch.randn(8, 19, device='cuda')

    print("\n5. Testing Forward Pass:")
    output = model(drug_batch, patient_batch)
    print(f"✅ Forward pass successful")
    print(f"   Input shape: drug={drug_batch.shape}, patient={patient_batch.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output device: {output.device}")

except Exception as e:
    print(f"❌ Model test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print("\nIf all tests passed ✅, you're ready to train!")
print("If any tests failed ❌, check the error messages above.")
