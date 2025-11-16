"""
Drug-Patient Interaction Pipeline with Quantum GNN

A comprehensive pipeline for predicting drug-patient interactions using
quantum graph neural networks (QGNN). This system combines molecular drug
features from protein-ligand binding data with patient characteristics to
predict treatment outcomes.

Main Components:
- DrugPatientDataProcessor: Load and process drug-patient data
- QuantumDrugPatientGNN: Quantum GNN model for interaction prediction
- DrugPatientTrainer: Training and evaluation utilities
- PatientFeatures: Patient data structure
- DrugPatientInteraction: Interaction data structure

Example:
    >>> from drug_patient_qgnn import (
    ...     DrugPatientDataProcessor,
    ...     QuantumDrugPatientGNN,
    ...     DrugPatientTrainer
    ... )
    >>>
    >>> # Load data
    >>> processor = DrugPatientDataProcessor(data_dir="othercode/data")
    >>> processor.load_protein_ligand_data(max_samples=100)
    >>> processor.create_synthetic_patient_data(n_patients=200)
    >>> processor.create_synthetic_interactions(interaction_rate=0.05)
    >>>
    >>> # Create model
    >>> graph = processor.graph
    >>> model = QuantumDrugPatientGNN(
    ...     drug_dim=graph.get_drug_features_matrix().shape[1],
    ...     patient_dim=graph.get_patient_features_matrix().shape[1],
    ...     num_qubits=4,
    ...     num_qlayers=2,
    ...     use_quantum=True
    ... )
    >>>
    >>> # Train
    >>> trainer = DrugPatientTrainer(model, learning_rate=0.001)
    >>> trainer.fit(graph, epochs=100, val_split=0.2)
"""

__version__ = "0.1.0"
__author__ = "Drug-Patient QGNN Team"
__license__ = "MIT"

# Import main classes
from .data_processing import (
    PatientFeatures,
    DrugPatientInteraction,
    BipartiteGraph,
    DrugPatientDataProcessor
)

from .model import (
    QuantumDrugPatientGNN,
    QuantumInteractionLayer,
    ClassicalInteractionLayer
)

from .trainer import DrugPatientTrainer

from .utils import (
    set_seed,
    print_model_summary,
    calculate_metrics,
    print_metrics,
    save_training_history,
    load_training_history,
    plot_training_history,
    get_device_info,
    print_device_info,
    estimate_training_time
)

# Define public API
__all__ = [
    # Data structures
    'PatientFeatures',
    'DrugPatientInteraction',
    'BipartiteGraph',

    # Main classes
    'DrugPatientDataProcessor',
    'QuantumDrugPatientGNN',
    'DrugPatientTrainer',

    # Model components (advanced usage)
    'QuantumInteractionLayer',
    'ClassicalInteractionLayer',

    # Utilities
    'set_seed',
    'print_model_summary',
    'calculate_metrics',
    'print_metrics',
    'save_training_history',
    'load_training_history',
    'plot_training_history',
    'get_device_info',
    'print_device_info',
    'estimate_training_time',
]
