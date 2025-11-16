"""
Tests for graph data structures and data processing.
"""

import sys
import os
import numpy as np
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from drug_patient_qgnn import (
    PatientFeatures,
    DrugPatientInteraction,
    BipartiteGraph,
    DrugPatientDataProcessor
)


def test_patient_features():
    """Test PatientFeatures data class."""
    print("Testing PatientFeatures...")

    patient = PatientFeatures(
        patient_id="P001",
        age=45.0,
        sex=1,
        weight=75.5,
        genetic_markers=np.array([0.1, 0.2, 0.3]),
        comorbidities=np.array([1, 0, 1]),
        lab_values=np.array([100, 1.1, 30, 25]),
        prior_medications=np.array([1, 0, 0, 1, 0])
    )

    vector = patient.to_vector()
    expected_len = 3 + 3 + 3 + 4 + 5  # age/sex/weight + genetic + comorbid + lab + meds
    assert len(vector) == expected_len, f"Expected length {expected_len}, got {len(vector)}"
    assert vector[0] == 45.0, "Age mismatch"
    assert vector[1] == 1, "Sex mismatch"
    assert vector[2] == 75.5, "Weight mismatch"

    print("  ✓ PatientFeatures working correctly")


def test_drug_patient_interaction():
    """Test DrugPatientInteraction data class."""
    print("Testing DrugPatientInteraction...")

    interaction = DrugPatientInteraction(
        drug_id="D001",
        patient_id="P001",
        efficacy=0.85,
        adverse_events=["nausea", "headache"],
        dose=100.0,
        duration=30.0,
        outcome=1
    )

    edge_features = interaction.to_edge_features()
    assert len(edge_features) == 5, "Edge features should have 5 elements"
    assert edge_features[0] == 0.85, "Efficacy mismatch"
    assert edge_features[1] == 100.0, "Dose mismatch"
    assert edge_features[2] == 30.0, "Duration mismatch"
    assert edge_features[3] == 1.0, "Outcome mismatch"
    assert edge_features[4] == 2.0, "Adverse event count mismatch"

    print("  ✓ DrugPatientInteraction working correctly")


def test_bipartite_graph():
    """Test BipartiteGraph construction."""
    print("Testing BipartiteGraph...")

    graph = BipartiteGraph()

    # Add drugs
    graph.add_drug("D001", np.array([1.0, 2.0, 3.0]))
    graph.add_drug("D002", np.array([4.0, 5.0, 6.0]))

    # Add patients
    patient1 = PatientFeatures(
        patient_id="P001",
        age=45.0,
        sex=1,
        weight=75.5,
        genetic_markers=np.array([0.1]),
        comorbidities=np.array([1]),
        lab_values=np.array([100]),
        prior_medications=np.array([1])
    )
    patient2 = PatientFeatures(
        patient_id="P002",
        age=62.0,
        sex=0,
        weight=68.0,
        genetic_markers=np.array([0.2]),
        comorbidities=np.array([0]),
        lab_values=np.array([95]),
        prior_medications=np.array([0])
    )

    graph.add_patient(patient1)
    graph.add_patient(patient2)

    # Add edges
    edge_features = np.array([0.85, 100.0, 30.0, 1.0, 2.0])
    graph.add_edge("D001", "P001", edge_features)
    graph.add_edge("D002", "P002", edge_features)

    # Check graph structure
    assert graph.num_drugs() == 2, "Should have 2 drugs"
    assert graph.num_patients() == 2, "Should have 2 patients"
    assert graph.num_edges() == 2, "Should have 2 edges"

    # Check feature matrices
    drug_matrix = graph.get_drug_features_matrix()
    assert drug_matrix.shape == (2, 3), f"Drug matrix shape mismatch: {drug_matrix.shape}"

    patient_matrix = graph.get_patient_features_matrix()
    assert patient_matrix.shape[0] == 2, "Should have 2 patient rows"

    # Check edge index
    edge_index, edge_feats = graph.get_edge_index()
    assert edge_index.shape == (2, 2), f"Edge index shape mismatch: {edge_index.shape}"
    assert edge_feats.shape == (2, 5), f"Edge features shape mismatch: {edge_feats.shape}"

    # Check labels
    labels = graph.get_edge_labels()
    assert len(labels) == 2, "Should have 2 labels"
    assert all(labels == 1.0), "All outcomes should be 1.0"

    print("  ✓ BipartiteGraph working correctly")


def test_data_processor_synthetic():
    """Test DrugPatientDataProcessor with synthetic data."""
    print("Testing DrugPatientDataProcessor (synthetic)...")

    processor = DrugPatientDataProcessor(seed=42)

    # Create synthetic drugs (since we likely don't have real PDB data)
    for i in range(10):
        processor.graph.add_drug(f"drug_{i}", np.random.randn(11))

    # Create synthetic patients
    processor.create_synthetic_patient_data(n_patients=20)

    # Create interactions
    processor.create_synthetic_interactions(interaction_rate=0.1)

    # Check statistics
    stats = processor.get_statistics()
    assert stats['num_drugs'] == 10, "Should have 10 drugs"
    assert stats['num_patients'] == 20, "Should have 20 patients"
    assert stats['num_interactions'] > 0, "Should have some interactions"

    print(f"  Created {stats['num_interactions']} interactions")
    print("  ✓ DrugPatientDataProcessor working correctly")


def test_save_load_graph():
    """Test saving and loading graph."""
    print("Testing save/load graph...")

    processor = DrugPatientDataProcessor(seed=42)

    # Create small graph
    for i in range(5):
        processor.graph.add_drug(f"drug_{i}", np.random.randn(11))
    processor.create_synthetic_patient_data(n_patients=10)
    processor.create_synthetic_interactions(interaction_rate=0.2)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name

    try:
        processor.save_graph(temp_path)

        # Load into new processor
        new_processor = DrugPatientDataProcessor(seed=42)
        new_processor.load_graph(temp_path)

        # Check that graphs match
        assert new_processor.graph.num_drugs() == 5, "Drug count mismatch"
        assert new_processor.graph.num_patients() == 10, "Patient count mismatch"
        assert new_processor.graph.num_edges() == processor.graph.num_edges(), "Edge count mismatch"

        print("  ✓ Save/load graph working correctly")

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_all_tests():
    """Run all graph tests."""
    print("\n" + "=" * 60)
    print("Running Graph Tests")
    print("=" * 60 + "\n")

    test_patient_features()
    test_drug_patient_interaction()
    test_bipartite_graph()
    test_data_processor_synthetic()
    test_save_load_graph()

    print("\n" + "=" * 60)
    print("All Graph Tests Passed! ✓")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_all_tests()
