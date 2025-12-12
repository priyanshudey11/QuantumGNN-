"""
Quick Start Script for Drug-Patient Interaction Pipeline

This script demonstrates how to use your existing PDB/protein-ligand data
to create a drug-patient interaction prediction model using Quantum GNN.

Usage:
    python run_drug_patient_pipeline.py

Author: Generated with Claude Code
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Import pipeline components
from drug_patient_quantum_gnn_pipeline import (
    DrugPatientDataProcessor,
    QuantumDrugPatientGNN,
    DrugPatientTrainer,
    PatientFeatures,
    DrugPatientInteraction
)


def quick_demo():
    """
    Quick demonstration using your existing data
    """
    print("\n" + "="*70)
    print("DRUG-PATIENT INTERACTION PIPELINE - QUICK START")
    print("="*70 + "\n")

    # Configuration
    DATA_DIR = "/Users/priyanshudey/Code/Qunatum/othercode/data"
    MAX_DRUGS = 50  # Limit for quick demo (remove for full dataset)
    N_PATIENTS = 100
    INTERACTION_RATE = 0.05  # 5% of drug-patient pairs have interactions

    # Model configuration
    NUM_QUBITS = 8
    NUM_QLAYERS = 2
    HIDDEN_DIM = 64
    LEARNING_RATE = 0.001
    EPOCHS = 50

    print(f"Configuration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Max drugs to load: {MAX_DRUGS}")
    print(f"  Number of patients: {N_PATIENTS}")
    print(f"  Quantum qubits: {NUM_QUBITS}")
    print(f"  Training epochs: {EPOCHS}\n")

    # Step 1: Initialize data processor
    print("Step 1: Initializing data processor...")
    processor = DrugPatientDataProcessor(data_dir=DATA_DIR)

    # Step 2: Load drug features from your PDB descriptor files
    print("\nStep 2: Loading drug features from protein-ligand binding data...")
    drug_descriptors = processor.load_protein_ligand_data(max_samples=MAX_DRUGS)

    if len(drug_descriptors) == 0:
        print("\nERROR: No drug descriptor files found!")
        print(f"   Please ensure {DATA_DIR} contains *_descriptors_3d.csv files")
        print("   These files should be in subdirectories like:")
        print("     data/1a0n/results/1a0n--A--P27986__Repair-H_descriptors_3d.csv")
        return False

    print(f"✓ Loaded {len(drug_descriptors)} drug compounds")

    # Step 3: Create patient data
    print("\nStep 3: Creating patient profiles...")
    print("   NOTE: Using synthetic patient data for demonstration")
    print("   Replace this with real patient data from your EHR/clinical database")
    processor.create_synthetic_patient_data(n_patients=N_PATIENTS)
    print(f"✓ Created {N_PATIENTS} patient profiles")

    # Step 4: Create drug-patient interactions
    print("\nStep 4: Creating drug-patient interaction records...")
    print("   NOTE: Using synthetic interactions for demonstration")
    print("   Replace this with real clinical outcomes data")
    processor.create_synthetic_interactions(interaction_rate=INTERACTION_RATE)

    n_interactions = len(processor.graph.interactions)
    if n_interactions == 0:
        print("\nERROR: No interactions created!")
        return False

    print(f"✓ Created {n_interactions} drug-patient interactions")

    # Step 5: Prepare graph
    graph = processor.graph
    drug_features = graph.get_drug_features_matrix()
    patient_features = graph.get_patient_features_matrix()

    print(f"\nGraph statistics:")
    print(f"  Number of drugs: {len(graph.drugs)}")
    print(f"  Number of patients: {len(graph.patients)}")
    print(f"  Number of interactions: {len(graph.interactions)}")
    print(f"  Drug feature dimension: {drug_features.shape[1]}")
    print(f"  Patient feature dimension: {patient_features.shape[1]}")

    # Step 6: Save processed graph
    print("\nStep 5: Saving processed graph...")
    graph_file = "drug_patient_graph.pkl"
    processor.save_graph(graph_file)
    print(f"✓ Graph saved to {graph_file}")

    # Step 7: Initialize quantum GNN model
    print("\nStep 6: Initializing Quantum GNN model...")

    drug_dim = drug_features.shape[1]
    patient_dim = patient_features.shape[1]

    try:
        model = QuantumDrugPatientGNN(
            drug_dim=drug_dim,
            patient_dim=patient_dim,
            hidden_dim=HIDDEN_DIM,
            num_qubits=NUM_QUBITS,
            num_qlayers=NUM_QLAYERS,
            use_quantum=True
        )

        # Check if quantum circuits are available
        if hasattr(model, 'qlayer'):
            print(f"✓ Quantum GNN initialized with {NUM_QUBITS} qubits")
            print(f"  Quantum device: default.qubit")
        else:
            print(f"✓ Classical GNN initialized (PennyLane not available)")

    except Exception as e:
        print(f" Error initializing model: {e}")
        return False

    # Step 8: Train model
    print(f"\nStep 7: Training model for {EPOCHS} epochs...")
    print("  (This may take a few minutes...)\n")

    trainer = DrugPatientTrainer(model, learning_rate=LEARNING_RATE, device='cpu')

    try:
        train_losses, val_losses = trainer.fit(
            graph,
            epochs=EPOCHS,
            val_split=0.2
        )
        print("\n✓ Training complete!")

    except Exception as e:
        print(f"\n Training error: {e}")
        return False

    # Step 9: Save trained model
    print("\nStep 8: Saving trained model...")
    model_file = "quantum_drug_patient_model.pt"
    torch.save(model.state_dict(), model_file)
    print(f"✓ Model saved to {model_file}")

    # Step 10: Test prediction on a sample
    print("\nStep 9: Testing prediction on sample drug-patient pair...")
    model.eval()

    # Get first drug and first patient
    sample_drug_features = drug_features[0:1]  # (1, drug_dim)
    sample_patient_features = patient_features[0:1]  # (1, patient_dim)

    with torch.no_grad():
        prediction = model(sample_drug_features, sample_patient_features)
        probability = prediction.item()

    print(f"✓ Sample prediction:")
    print(f"  Drug: {list(graph.drugs.keys())[0]}")
    print(f"  Patient: {list(graph.patients.keys())[0]}")
    print(f"  Predicted success probability: {probability:.2%}")

    if probability > 0.5:
        print(f"  → Model predicts SUCCESSFUL treatment")
    else:
        print(f"  → Model predicts UNSUCCESSFUL treatment")

    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  1. {graph_file} - Processed drug-patient graph")
    print(f"  2. {model_file} - Trained quantum GNN model")

    print("\nNext steps:")
    print("  1. Replace synthetic patient data with real clinical data")
    print("  2. Replace synthetic interactions with real outcomes data")
    print("  3. Tune hyperparameters (num_qubits, learning_rate, epochs)")
    print("  4. Validate on held-out test set")
    print("  5. Deploy for drug-patient matching")

    print("\nFor more details, see: DRUG_PATIENT_PIPELINE_README.md")
    print()

    return True


def load_and_predict_demo():
    """
    Demonstrate loading a saved model and making predictions
    """
    print("\n" + "="*70)
    print("LOADING SAVED MODEL FOR PREDICTION")
    print("="*70 + "\n")

    graph_file = "drug_patient_graph.pkl"
    model_file = "quantum_drug_patient_model.pt"

    # Check if files exist
    if not Path(graph_file).exists():
        print(f" Graph file not found: {graph_file}")
        print("   Run the quick demo first to generate data and train model")
        return False

    if not Path(model_file).exists():
        print(f" Model file not found: {model_file}")
        print("   Run the quick demo first to train model")
        return False

    # Load graph
    print(f"Loading graph from {graph_file}...")
    processor = DrugPatientDataProcessor(data_dir=".")
    graph = processor.load_graph(graph_file)

    # Get dimensions
    drug_features = graph.get_drug_features_matrix()
    patient_features = graph.get_patient_features_matrix()
    drug_dim = drug_features.shape[1]
    patient_dim = patient_features.shape[1]

    # Initialize model
    print(f"Initializing model architecture...")
    model = QuantumDrugPatientGNN(
        drug_dim=drug_dim,
        patient_dim=patient_dim,
        num_qubits=8,
        use_quantum=True
    )

    # Load weights
    print(f"Loading model weights from {model_file}...")
    model.load_state_dict(torch.load(model_file))
    model.eval()

    print("✓ Model loaded successfully!\n")

    # Make predictions on all drug-patient pairs
    print("Making predictions for all drug-patient combinations...")

    n_drugs = len(graph.drugs)
    n_patients = len(graph.patients)

    drug_ids = list(graph.drugs.keys())
    patient_ids = list(graph.patients.keys())

    # Predict for first 5 drugs and first 5 patients
    print(f"\nTop predictions (first 5 drugs × first 5 patients):\n")

    predictions = []
    with torch.no_grad():
        for i in range(min(5, n_drugs)):
            for j in range(min(5, n_patients)):
                drug_feat = drug_features[i:i+1]
                patient_feat = patient_features[j:j+1]
                prob = model(drug_feat, patient_feat).item()

                predictions.append({
                    'drug': drug_ids[i],
                    'patient': patient_ids[j],
                    'probability': prob
                })

    # Sort by probability
    predictions.sort(key=lambda x: x['probability'], reverse=True)

    # Show top 10
    print("Top 10 predicted successful drug-patient matches:\n")
    for idx, pred in enumerate(predictions[:10], 1):
        drug_short = pred['drug'][:40] + "..." if len(pred['drug']) > 40 else pred['drug']
        print(f"{idx:2d}. {pred['probability']:.2%} - {drug_short} → {pred['patient']}")

    print("\n✓ Prediction demo complete!\n")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DRUG-PATIENT INTERACTION QUANTUM GNN PIPELINE")
    print("="*70)

    if len(sys.argv) > 1 and sys.argv[1] == "--predict":
        # Load and predict mode
        success = load_and_predict_demo()
    else:
        # Training mode
        print("\nMode: Training")
        print("(Use --predict flag to load saved model and make predictions)\n")
        success = quick_demo()

    if success:
        print("✅ SUCCESS!")
    else:
        print("\n❌ Pipeline encountered errors")
        print("   Check the error messages above for details")

    print()
