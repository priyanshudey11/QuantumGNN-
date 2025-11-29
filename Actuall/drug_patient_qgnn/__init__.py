"""
Drug-Protein Interaction Pipeline with Quantum GNN

A comprehensive pipeline for predicting drug-protein interactions using
quantum graph neural networks (QGNN). This system combines molecular ligand
features with protein pocket descriptors from binding data to predict
ligand-protein binding and interaction types.

Main Components:
- DrugProteinDataProcessor: Load and process drug-protein data
- QuantumDrugProteinGNN: Quantum GNN model for interaction prediction
- DrugProteinTrainer: Training and evaluation utilities
- ProteinPocketFeatures: Protein pocket data structure
- DrugProteinInteraction: Interaction data structure

Example:
    >>> from drug_patient_qgnn import (
    ...     DrugProteinDataProcessor,
    ...     QuantumDrugProteinGNN,
    ...     DrugProteinTrainer
    ... )
    >>>
    >>> # Load data
    >>> processor = DrugProteinDataProcessor(data_dir="/media/priyanshu/SD/othercode/data")
    >>> processor.load_protein_ligand_data(max_samples=100)
    >>> processor.create_synthetic_ligand_data(n_ligands=100)
    >>> processor.create_synthetic_interactions(interaction_rate=0.05)
    >>>
    >>> # Create model
    >>> graph = processor.graph
    >>> model = QuantumDrugProteinGNN(
    ...     ligand_dim=graph.get_ligand_features_matrix().shape[1],
    ...     pocket_dim=graph.get_pocket_features_matrix().shape[1],
    ...     num_qubits=4,
    ...     num_qlayers=2,
    ...     use_quantum=True
    ... )
    >>>
    >>> # Train
    >>> trainer = DrugProteinTrainer(model, learning_rate=0.001)
    >>> trainer.fit(graph, epochs=100, val_split=0.2)
"""

__version__ = "0.2.0"
__author__ = "Drug-Protein QGNN Team"
__license__ = "MIT"

# Import main classes
from .data_processing import (
    ProteinPocketFeatures,
    LigandFeatures,
    DrugProteinInteraction,
    BipartiteGraph,
    DrugProteinDataProcessor
)

from .model import (
    QuantumDrugProteinGNN,
    QuantumInteractionLayer,
    ClassicalInteractionLayer
)

from .trainer import (
    DrugProteinTrainer
)

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
    # New data structures (drug-protein paradigm)
    'ProteinPocketFeatures',
    'LigandFeatures',
    'DrugProteinInteraction',
    'BipartiteGraph',

    # Main classes (new names)
    'DrugProteinDataProcessor',
    'QuantumDrugProteinGNN',
    'DrugProteinTrainer',

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

    # Backward compatibility (deprecated)
    'PatientFeatures',
    'DrugPatientInteraction',
    'DrugPatientDataProcessor',
    'QuantumDrugPatientGNN',
    'DrugPatientTrainer',
]

# Create backward compatibility aliases (deprecated, will be removed in v1.0)
# These map old patient-based names to new protein-based implementations
PatientFeatures = ProteinPocketFeatures
DrugPatientInteraction = DrugProteinInteraction
DrugPatientDataProcessor = DrugProteinDataProcessor
QuantumDrugPatientGNN = QuantumDrugProteinGNN
DrugPatientTrainer = DrugProteinTrainer
