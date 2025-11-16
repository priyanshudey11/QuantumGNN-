"""
Example script for running the Drug-Patient QGNN Pipeline.

This script demonstrates:
1. Loading drug data from PDB descriptors
2. Creating synthetic patient data
3. Creating synthetic interactions
4. Training a quantum GNN model
5. Evaluating and saving the model
"""

import sys
import os
import torch
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from drug_patient_qgnn import (
    DrugPatientDataProcessor,
    QuantumDrugPatientGNN,
    DrugPatientTrainer,
    set_seed,
    print_model_summary,
    print_device_info,
    plot_training_history
)


def main():
    parser = argparse.ArgumentParser(
        description='Run Drug-Patient QGNN Pipeline'
    )

    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        default='othercode/data',
        help='Directory containing PDB data'
    )
    parser.add_argument(
        '--max_drugs',
        type=int,
        default=None,
        help='Maximum number of drugs to load (None = all)'
    )
    parser.add_argument(
        '--n_patients',
        type=int,
        default=200,
        help='Number of synthetic patients to generate'
    )
    parser.add_argument(
        '--interaction_rate',
        type=float,
        default=0.05,
        help='Fraction of drug-patient pairs to create interactions'
    )

    # Model arguments
    parser.add_argument(
        '--num_qubits',
        type=int,
        default=4,
        help='Number of qubits per side (total = 2 * num_qubits)'
    )
    parser.add_argument(
        '--num_qlayers',
        type=int,
        default=2,
        help='Number of variational quantum layers'
    )
    parser.add_argument(
        '--use_quantum',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='Use quantum circuit (True) or classical fallback (False)'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Validation split fraction'
    )

    # Other arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cpu, cuda, mps, or None for auto)'
    )
    parser.add_argument(
        '--save_model',
        type=str,
        default='trained_model.pt',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--save_graph',
        type=str,
        default=None,
        help='Path to save processed graph data'
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    print("\n" + "=" * 70)
    print("Drug-Patient Interaction Pipeline with Quantum GNN")
    print("=" * 70)

    # Print device information
    print_device_info()

    # ========================================================================
    # Step 1: Load and Process Data
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading and Processing Data")
    print("=" * 70 + "\n")

    processor = DrugPatientDataProcessor(data_dir=args.data_dir, seed=args.seed)

    # Load drug features from PDB descriptors
    print(f"Loading drug data from: {args.data_dir}")
    n_drugs_loaded = processor.load_protein_ligand_data(max_samples=args.max_drugs)

    # If no real drug data found, create synthetic drug data
    if n_drugs_loaded == 0:
        print("\nNo PDB data found. Creating synthetic drug data...")
        # Create 50 synthetic drugs with 11 features each
        for i in range(50):
            drug_id = f"synthetic_drug_{i}"
            features = torch.randn(11).numpy()  # Match typical descriptor count
            processor.graph.add_drug(drug_id, features)
        print(f"Created {processor.graph.num_drugs()} synthetic drug nodes")

    # Create synthetic patient data
    print(f"\nGenerating {args.n_patients} synthetic patients...")
    processor.create_synthetic_patient_data(n_patients=args.n_patients)

    # Create synthetic interactions
    print(f"\nGenerating interactions (rate={args.interaction_rate})...")
    processor.create_synthetic_interactions(interaction_rate=args.interaction_rate)

    # Print statistics
    stats = processor.get_statistics()
    print("\nDataset Statistics:")
    print(f"  Drugs:        {stats['num_drugs']}")
    print(f"  Patients:     {stats['num_patients']}")
    print(f"  Interactions: {stats['num_interactions']}")
    print(f"  Drug dim:     {stats['drug_feature_dim']}")
    print(f"  Patient dim:  {stats['patient_feature_dim']}")
    print(f"  Positive rate: {stats.get('positive_rate', 0):.2%}")

    # Save graph if requested
    if args.save_graph:
        processor.save_graph(args.save_graph)

    # ========================================================================
    # Step 2: Create Model
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Creating Model")
    print("=" * 70 + "\n")

    graph = processor.graph
    drug_dim = stats['drug_feature_dim']
    patient_dim = stats['patient_feature_dim']

    model = QuantumDrugPatientGNN(
        drug_dim=drug_dim,
        patient_dim=patient_dim,
        num_qubits=args.num_qubits,
        num_qlayers=args.num_qlayers,
        use_quantum=args.use_quantum
    )

    print_model_summary(model, drug_dim, patient_dim)

    # ========================================================================
    # Step 3: Train Model
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Training Model")
    print("=" * 70 + "\n")

    trainer = DrugPatientTrainer(
        model,
        learning_rate=args.learning_rate,
        device=args.device
    )

    history = trainer.fit(
        graph,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        verbose=2
    )

    # ========================================================================
    # Step 4: Save Model and Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Saving Results")
    print("=" * 70 + "\n")

    # Save model
    torch.save(model.state_dict(), args.save_model)
    print(f"Model saved to: {args.save_model}")

    # Save training history plot
    plot_path = args.save_model.replace('.pt', '_history.png')
    try:
        plot_training_history(history, save_path=plot_path)
    except Exception as e:
        print(f"Could not save training plot: {e}")

    # ========================================================================
    # Step 5: Test Prediction
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Testing Prediction")
    print("=" * 70 + "\n")

    # Get sample drug and patient
    drug_features = torch.tensor(
        graph.get_drug_features_matrix()[:1],
        dtype=torch.float32
    ).to(trainer.device)

    patient_features = torch.tensor(
        graph.get_patient_features_matrix()[:1],
        dtype=torch.float32
    ).to(trainer.device)

    model.eval()
    with torch.no_grad():
        outcome_prob = model(drug_features, patient_features)
        print(f"Sample prediction - Success probability: {outcome_prob.item():.2%}")

    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
