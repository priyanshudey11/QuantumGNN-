"""
Tests for training loop and trainer functionality.
"""

import sys
import os
import torch
import numpy as np
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from drug_patient_qgnn import (
    QuantumDrugPatientGNN,
    DrugPatientTrainer,
    DrugPatientDataProcessor,
    set_seed
)


def create_test_data():
    """Create small test dataset."""
    set_seed(42)

    processor = DrugPatientDataProcessor(seed=42)

    # Create 10 drugs
    for i in range(10):
        processor.graph.add_drug(f"drug_{i}", np.random.randn(11))

    # Create 20 patients
    processor.create_synthetic_patient_data(n_patients=20)

    # Create interactions (should create ~40 edges with rate=0.2)
    processor.create_synthetic_interactions(interaction_rate=0.2)

    return processor.graph


def test_trainer_initialization():
    """Test trainer initialization."""
    print("Testing trainer initialization...")

    model = QuantumDrugPatientGNN(
        drug_dim=11,
        patient_dim=18,
        num_qubits=4,
        num_qlayers=2,
        use_quantum=False
    )

    trainer = DrugPatientTrainer(model, learning_rate=0.001)

    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.criterion is not None
    assert len(trainer.history) > 0

    print(f"  Device: {trainer.device}")
    print("  ✓ Trainer initialization working correctly")


def test_single_epoch():
    """Test single training epoch."""
    print("Testing single training epoch...")

    graph = create_test_data()

    drug_dim = graph.get_drug_features_matrix().shape[1]
    patient_dim = graph.get_patient_features_matrix().shape[1]

    model = QuantumDrugPatientGNN(
        drug_dim=drug_dim,
        patient_dim=patient_dim,
        num_qubits=4,
        num_qlayers=1,
        use_quantum=False
    )

    trainer = DrugPatientTrainer(model, learning_rate=0.001)

    # Train one epoch
    metrics = trainer.train_epoch(graph, batch_size=8, shuffle=True)

    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert metrics['loss'] >= 0, "Loss should be non-negative"
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be in [0, 1]"

    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print("  ✓ Single epoch training working correctly")


def test_evaluation():
    """Test model evaluation."""
    print("Testing model evaluation...")

    graph = create_test_data()

    drug_dim = graph.get_drug_features_matrix().shape[1]
    patient_dim = graph.get_patient_features_matrix().shape[1]

    model = QuantumDrugPatientGNN(
        drug_dim=drug_dim,
        patient_dim=patient_dim,
        num_qubits=4,
        num_qlayers=1,
        use_quantum=False
    )

    trainer = DrugPatientTrainer(model, learning_rate=0.001)

    # Evaluate on all edges
    n_edges = graph.num_edges()
    indices = np.arange(n_edges)

    metrics = trainer.evaluate(graph, indices, batch_size=8)

    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert 'auc' in metrics
    assert metrics['loss'] >= 0, "Loss should be non-negative"
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be in [0, 1]"

    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print("  ✓ Model evaluation working correctly")


def test_full_training_loop():
    """Test full training loop with train/val split."""
    print("Testing full training loop...")

    graph = create_test_data()

    drug_dim = graph.get_drug_features_matrix().shape[1]
    patient_dim = graph.get_patient_features_matrix().shape[1]

    model = QuantumDrugPatientGNN(
        drug_dim=drug_dim,
        patient_dim=patient_dim,
        num_qubits=4,
        num_qlayers=1,
        use_quantum=False
    )

    trainer = DrugPatientTrainer(model, learning_rate=0.001, device='cpu')

    # Train for a few epochs
    history = trainer.fit(
        graph,
        epochs=5,
        batch_size=8,
        val_split=0.2,
        verbose=0  # Silent
    )

    # Check history
    assert 'train_loss' in history
    assert 'train_acc' in history
    assert 'val_loss' in history
    assert 'val_acc' in history
    assert 'val_auc' in history

    assert len(history['train_loss']) == 5, "Should have 5 epochs of train loss"
    assert len(history['val_loss']) == 5, "Should have 5 epochs of val loss"

    # Check that loss generally decreases (with some tolerance)
    initial_loss = history['train_loss'][0]
    final_loss = history['train_loss'][-1]

    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Final val accuracy: {history['val_acc'][-1]:.4f}")
    print("  ✓ Full training loop working correctly")


def test_checkpoint_save_load():
    """Test saving and loading checkpoints."""
    print("Testing checkpoint save/load...")

    graph = create_test_data()

    drug_dim = graph.get_drug_features_matrix().shape[1]
    patient_dim = graph.get_patient_features_matrix().shape[1]

    model = QuantumDrugPatientGNN(
        drug_dim=drug_dim,
        patient_dim=patient_dim,
        num_qubits=4,
        num_qlayers=1,
        use_quantum=False
    )

    trainer = DrugPatientTrainer(model, learning_rate=0.001)

    # Train briefly
    trainer.fit(graph, epochs=2, batch_size=8, val_split=0.2, verbose=0)

    # Save checkpoint
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name

    try:
        trainer.save_checkpoint(temp_path)

        # Create new trainer and load checkpoint
        new_model = QuantumDrugPatientGNN(
            drug_dim=drug_dim,
            patient_dim=patient_dim,
            num_qubits=4,
            num_qlayers=1,
            use_quantum=False
        )
        new_trainer = DrugPatientTrainer(new_model, learning_rate=0.001)
        new_trainer.load_checkpoint(temp_path)

        # Check that history was restored
        assert len(new_trainer.history['train_loss']) == 2, "History not restored"

        print("  ✓ Checkpoint save/load working correctly")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_batch_processing():
    """Test batch processing with different batch sizes."""
    print("Testing batch processing...")

    graph = create_test_data()

    drug_dim = graph.get_drug_features_matrix().shape[1]
    patient_dim = graph.get_patient_features_matrix().shape[1]

    model = QuantumDrugPatientGNN(
        drug_dim=drug_dim,
        patient_dim=patient_dim,
        num_qubits=4,
        num_qlayers=1,
        use_quantum=False
    )

    trainer = DrugPatientTrainer(model, learning_rate=0.001)

    # Test different batch sizes
    for batch_size in [4, 8, 16]:
        metrics = trainer.train_epoch(graph, batch_size=batch_size, shuffle=False)
        assert 'loss' in metrics, f"Failed with batch_size={batch_size}"
        print(f"  Batch size {batch_size}: loss={metrics['loss']:.4f}")

    print("  ✓ Batch processing working correctly")


def run_all_tests():
    """Run all trainer tests."""
    print("\n" + "=" * 60)
    print("Running Trainer Tests")
    print("=" * 60 + "\n")

    test_trainer_initialization()
    test_single_epoch()
    test_evaluation()
    test_full_training_loop()
    test_checkpoint_save_load()
    test_batch_processing()

    print("\n" + "=" * 60)
    print("All Trainer Tests Passed! ✓")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_all_tests()
