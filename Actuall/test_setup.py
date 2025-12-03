#!/usr/bin/env python
"""
Quick test to verify the must-run experiment setup is correct.
"""

import sys
import torch
import pennylane as qml

print("="*70)
print("MUST-RUN EXPERIMENT - SETUP VERIFICATION")
print("="*70)

# Test 1: PyTorch + CUDA
print("\n[1] Testing PyTorch + CUDA...")
print(f"    PyTorch version: {torch.__version__}")
print(f"    CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    CUDA version: {torch.version.cuda}")
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("    ✓ CUDA is ready")
else:
    print("    ⚠ CUDA not available - will use CPU")

# Test 2: PennyLane
print("\n[2] Testing PennyLane...")
print(f"    PennyLane version: {qml.__version__}")
print("    ✓ PennyLane imported successfully")

# Test 3: lightning.gpu
print("\n[3] Testing lightning.gpu device...")
try:
    dev = qml.device('lightning.gpu', wires=8)
    print(f"    Device: {dev.name}")
    print(f"    Wires: {dev.num_wires}")
    print("    ✓ lightning.gpu is AVAILABLE and ready!")
except Exception as e:
    print(f"    ⚠ lightning.gpu not available: {e}")
    print("    Will fallback to lightning.qubit")

# Test 4: Model import
print("\n[4] Testing model import...")
try:
    from ligand_pocket_qgnn.model import LigandPocketQGNN
    print("    ✓ LigandPocketQGNN imported successfully")
except Exception as e:
    print(f"    ✗ Failed to import model: {e}")
    sys.exit(1)

# Test 5: Create test model
print("\n[5] Testing model creation...")
try:
    test_model = LigandPocketQGNN(
        ligand_in_dim=10,
        pocket_in_dim=19,
        hidden_dim=64,
        n_qubits=8,
        n_qlayers=4,
        use_quantum=True,
        quantum_device='lightning.gpu'
    )
    print(f"    Total parameters: {sum(p.numel() for p in test_model.parameters()):,}")
    print("    ✓ Model created successfully")
except Exception as e:
    print(f"    ✗ Failed to create model: {e}")
    sys.exit(1)

# Test 6: Forward pass
print("\n[6] Testing forward pass...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_model = test_model.to(device)

    # Create dummy batch
    x = torch.randn(10, 10).to(device)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(device)
    batch = torch.zeros(10, dtype=torch.long).to(device)
    pocket = torch.randn(1, 19).to(device)

    output = test_model(x, edge_index, batch, pocket)
    print(f"    Output shape: {output.shape}")
    print(f"    Output value: {output.item():.4f}")
    print("    ✓ Forward pass successful")
except Exception as e:
    print(f"    ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED - READY TO RUN EXPERIMENT!")
print("="*70)
print("\nNext steps:")
print("  1. Open: compare_ligand_pocket_quantum_vs_classical.ipynb")
print("  2. Run all cells")
print("  3. Monitor GPU usage with: nvidia-smi -l 1")
print("\nExpected configuration:")
print("  • Qubits: 8")
print("  • Depth: 4")
print("  • Device: lightning.gpu")
print("  • Batch size: 512")
print("  • Workers: 8")
print("\nGood luck! 🚀")
