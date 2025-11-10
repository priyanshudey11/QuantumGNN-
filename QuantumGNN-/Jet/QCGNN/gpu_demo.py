#!/usr/bin/env python3
"""
GPU Acceleration Demo for QCGNN

This script demonstrates how to use CUDA and Metal (MPS) acceleration
with the QCGNN quantum neural network models.

Requirements:
- For CUDA: NVIDIA GPU with CUDA support
- For Metal: Apple Silicon Mac (M1/M2/M3) or Intel Mac with Metal support
- PennyLane with appropriate plugins installed

Usage:
    python gpu_demo.py --mode CUDA       # For NVIDIA GPUs
    python gpu_demo.py --mode Metal      # For Apple GPUs
    python gpu_demo.py --mode Lightning_GPU  # For Lightning GPU (CUDA)
"""

import argparse
import torch
import yaml
from source.models.qcgnn import QuantumRotQCGNN
import pennylane as qml

def check_gpu_availability():
    """Check available GPU devices."""
    print("=== GPU Availability Check ===")

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        print(f"  CUDA devices: {torch.cuda.device_count()}")
    else:
        print("✗ CUDA not available")

    # Check Metal (MPS)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✓ Metal Performance Shaders (MPS) available")
    else:
        print("✗ Metal Performance Shaders (MPS) not available")

    print()

def test_quantum_device(mode: str):
    """Test quantum device creation and basic operations."""
    print(f"=== Testing {mode} Quantum Device ===")

    # Load configuration
    with open("configs/ibmq.yaml", 'r') as file:
        config = yaml.safe_load(file)

    device_config = config['Device'][mode]
    print(f"Device config: {device_config}")

    try:
        # Create a simple quantum model
        model = QuantumRotQCGNN(
            num_ir_qubits=2,
            num_nr_qubits=4,
            num_layers=2,
            num_reupload=1,
            vqc_ansatz=qml.StronglyEntanglingLayers,
            score_dim=1,
            **device_config
        )

        # Test with dummy data
        batch_size = 4
        num_particles = 4
        num_features = 3  # (pt, eta, phi)
        x = torch.randn(batch_size, num_particles, num_features)

        print(f"Input shape: {x.shape}")

        # Forward pass
        with torch.no_grad():
            output = model(x)

        print(f"Output shape: {output.shape}")
        print(f"✓ {mode} device test successful!")

        # Get device info
        qml_device = model.phi.qml_device
        print(f"Quantum device: {qml_device}")
        if hasattr(qml_device, 'capabilities') and callable(qml_device.capabilities):
            print(f"Device capabilities: {qml_device.capabilities()}")

    except Exception as e:
        print(f"✗ {mode} device test failed: {e}")

    print()

def main():
    parser = argparse.ArgumentParser(description='GPU Acceleration Demo for QCGNN')
    parser.add_argument('--mode', type=str, choices=['CUDA', 'Metal', 'Lightning_GPU', 'Simulator'],
                       default='Simulator', help='GPU acceleration mode')

    args = parser.parse_args()

    print("QCGNN GPU Acceleration Demo")
    print("=" * 40)

    check_gpu_availability()
    test_quantum_device(args.mode)


if __name__ == "__main__":
    main()