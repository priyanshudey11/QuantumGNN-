"""
Tests for model forward pass and quantum circuit.
"""

import sys
import os
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from drug_patient_qgnn import QuantumDrugPatientGNN


def test_classical_mode_forward():
    """Test forward pass in classical mode."""
    print("Testing classical mode forward pass...")

    drug_dim = 11
    patient_dim = 18
    batch_size = 4

    model = QuantumDrugPatientGNN(
        drug_dim=drug_dim,
        patient_dim=patient_dim,
        num_qubits=4,
        num_qlayers=2,
        use_quantum=False  # Classical mode
    )

    # Create random input
    drug_features = torch.randn(batch_size, drug_dim)
    patient_features = torch.randn(batch_size, patient_dim)

    # Forward pass
    output = model(drug_features, patient_features)

    # Check output shape
    assert output.shape == (batch_size, 1), f"Output shape mismatch: {output.shape}"

    # Check output range (should be probabilities in [0, 1])
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output not in [0, 1]"

    # Check gradient flow
    output.sum().backward()
    assert model.drug_encoder[0].weight.grad is not None, "No gradients for drug encoder"
    assert model.patient_encoder[0].weight.grad is not None, "No gradients for patient encoder"

    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("  ✓ Classical mode forward pass working correctly")


def test_quantum_mode_forward():
    """Test forward pass in quantum mode (if PennyLane available)."""
    print("Testing quantum mode forward pass...")

    try:
        import pennylane as qml
    except ImportError:
        print("  ⚠ PennyLane not available, skipping quantum mode test")
        return

    drug_dim = 11
    patient_dim = 18
    batch_size = 2  # Smaller batch for quantum (slower)

    model = QuantumDrugPatientGNN(
        drug_dim=drug_dim,
        patient_dim=patient_dim,
        num_qubits=4,
        num_qlayers=1,  # Fewer layers for speed
        use_quantum=True  # Quantum mode
    )

    # Create random input
    drug_features = torch.randn(batch_size, drug_dim)
    patient_features = torch.randn(batch_size, patient_dim)

    # Forward pass
    output = model(drug_features, patient_features)

    # Check output shape
    assert output.shape == (batch_size, 1), f"Output shape mismatch: {output.shape}"

    # Check output range
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output not in [0, 1]"

    # Check gradient flow
    output.sum().backward()
    assert model.drug_encoder[0].weight.grad is not None, "No gradients for drug encoder"
    assert model.interaction_layer.q_params.grad is not None, "No gradients for quantum params"

    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  Quantum params: {model.interaction_layer.q_params.numel()}")
    print("  ✓ Quantum mode forward pass working correctly")


def test_model_prediction():
    """Test prediction methods."""
    print("Testing model prediction methods...")

    model = QuantumDrugPatientGNN(
        drug_dim=11,
        patient_dim=18,
        num_qubits=4,
        num_qlayers=2,
        use_quantum=False
    )

    drug_features = torch.randn(10, 11)
    patient_features = torch.randn(10, 18)

    # Test predict_batch
    probs, preds = model.predict_batch(drug_features, patient_features, threshold=0.5)

    assert probs.shape == (10,), f"Probs shape mismatch: {probs.shape}"
    assert preds.shape == (10,), f"Preds shape mismatch: {preds.shape}"
    assert torch.all((preds == 0) | (preds == 1)), "Predictions should be binary"

    print(f"  Predictions: {preds.tolist()}")
    print(f"  Mean probability: {probs.mean().item():.4f}")
    print("  ✓ Prediction methods working correctly")


def test_model_info():
    """Test model information methods."""
    print("Testing model information methods...")

    model = QuantumDrugPatientGNN(
        drug_dim=11,
        patient_dim=18,
        num_qubits=4,
        num_qlayers=2,
        use_quantum=False
    )

    # Get model info
    info = model.get_model_info()

    assert 'drug_dim' in info
    assert 'patient_dim' in info
    assert 'num_qubits' in info
    assert 'num_qlayers' in info
    assert 'use_quantum' in info
    assert 'num_parameters' in info

    assert info['drug_dim'] == 11
    assert info['patient_dim'] == 18
    assert info['num_qubits'] == 4
    assert info['num_qlayers'] == 2
    assert info['use_quantum'] == False

    # Get parameter count
    num_params = model.get_num_parameters()
    assert num_params > 0, "Model should have parameters"
    assert num_params == info['num_parameters']

    print(f"  Model info: {info}")
    print("  ✓ Model information methods working correctly")


def test_model_save_load():
    """Test model saving and loading."""
    print("Testing model save/load...")

    import tempfile

    model = QuantumDrugPatientGNN(
        drug_dim=11,
        patient_dim=18,
        num_qubits=4,
        num_qlayers=2,
        use_quantum=False
    )

    # Get initial output
    drug_features = torch.randn(2, 11)
    patient_features = torch.randn(2, 18)
    output1 = model(drug_features, patient_features)

    # Save model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name

    try:
        torch.save(model.state_dict(), temp_path)

        # Create new model and load weights
        new_model = QuantumDrugPatientGNN(
            drug_dim=11,
            patient_dim=18,
            num_qubits=4,
            num_qlayers=2,
            use_quantum=False
        )
        new_model.load_state_dict(torch.load(temp_path))

        # Get output from loaded model
        new_model.eval()
        model.eval()
        with torch.no_grad():
            output2 = new_model(drug_features, patient_features)

        # Outputs should match
        assert torch.allclose(output1, output2, atol=1e-6), "Loaded model outputs don't match"

        print("  ✓ Model save/load working correctly")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_all_tests():
    """Run all model tests."""
    print("\n" + "=" * 60)
    print("Running Model Tests")
    print("=" * 60 + "\n")

    test_classical_mode_forward()
    test_quantum_mode_forward()
    test_model_prediction()
    test_model_info()
    test_model_save_load()

    print("\n" + "=" * 60)
    print("All Model Tests Passed! ✓")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_all_tests()
