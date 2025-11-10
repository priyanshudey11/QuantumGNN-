"""
Quick training script to run all models with reduced dataset size for testing.
This script trains all 6 models from the QCGNN paper with 500 samples per class.
"""

import sys
import time
import yaml
import awkward as ak
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import pennylane as qml

# Local imports
from source.data.opendata import JetNetEvents
from source.data.datamodule import JetGraphDataModule, JetTorchDataModule
from source.models.qcgnn import QuantumRotQCGNN
from source.models.mpgnn import ClassicalMPGNN
from source.models.pfn import ParticleFlowNetwork
from source.models.part import ParticleTransformer
from source.models.pnet import ParticleNet
from source.training.litmodel import TorchLightningModule, GraphLightningModel
from source.training.loggers import csv_logger

print("=" * 80)
print("QCGNN Quick Training - All Models")
print("=" * 80)

# Configuration
dataset = 'JetNet'
random_seed = 42

# Load and modify config for quick testing
with open("configs/config.yaml", 'r') as file:
    config = yaml.safe_load(file)
    config['date'] = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    config['dataset'] = dataset

    # Quick test configuration: 500 samples per class
    config['Data']['num_train'] = 500
    config['Data']['num_valid'] = 100
    config['Data']['num_test'] = 100
    config['Train']['max_epochs'] = 10  # Reduced epochs for quick testing
    config['Settings']['use_wandb'] = False  # Disable wandb for simplicity
    config['Settings']['print_log'] = True   # Enable console logging

# Determine score dimension for 5-class classification
score_dim = config[dataset]['num_classes']

print(f"\n📊 Configuration:")
print(f"   Dataset: {dataset}")
print(f"   Training samples: {config['Data']['num_train']} per class")
print(f"   Validation samples: {config['Data']['num_valid']} per class")
print(f"   Test samples: {config['Data']['num_test']} per class")
print(f"   Max epochs: {config['Train']['max_epochs']}")
print(f"   Batch size: {config['Data']['batch_size']}")
print(f"   Max particles per jet: {config['Data']['max_num_ptcs']}")

# Check GPU availability
print(f"\n🖥️  Hardware:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / (1024**3)
    print(f"   GPU memory: {total_memory:.2f} GB")
else:
    print("   Using CPU (training will be slower)")

# Configure GPU memory limit
if torch.cuda.is_available():
    gpu_memory_fraction = config.get('Performance', {}).get('gpu_memory_fraction', 0.8)
    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=0)


def create_data_module(graph: bool, pi_scale: bool = False) -> L.LightningDataModule:
    """Create data module for training."""

    num_train = config['Data']['num_train']
    num_valid = config['Data']['num_valid']
    num_test = config['Data']['num_test']
    num_events = num_train + num_valid + num_test

    dataset_config = {}
    dataset_config.update(config['Data'])
    dataset_config.update(config[dataset])

    # Load JetNet dataset (5 particle types)
    channels = ['q', 'g', 't', 'w', 'z']
    events = [JetNetEvents(channel=channel, **dataset_config) for channel in channels]
    events = [_events.generate_events(num_events) for _events in events]

    # Create appropriate data module
    if graph:
        data_module = JetGraphDataModule(events, pi_scale=pi_scale, **dataset_config)
    else:
        data_module = JetTorchDataModule(events, **dataset_config)

    return data_module


def create_training_info(model: nn.Module, model_description: str, model_hparams: dict, lr: float) -> dict:
    """Create training information dictionary."""

    data_description = f"{dataset}_P{config['Data']['max_num_ptcs']}_N{config['Data']['num_train']}"
    group_rnd = '-'.join([model.__class__.__name__, model_description, data_description])
    name = f"{group_rnd}-{random_seed}"

    training_info = config.copy()
    training_info.update(model_hparams)
    training_info.update({
        'lr': lr,
        'name': name,
        'date': config['date'],
        'model': model.__class__.__name__,
        'group_rnd': group_rnd,
        'random_seed': random_seed,
        'data_description': data_description,
        'model_description': model_description,
    })

    return training_info


def create_lightning_model(model: nn.Module, graph: bool, lr: float) -> L.LightningModule:
    """Create lightning model wrapper."""

    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
    print_log = config['Settings']['print_log']

    if graph:
        return GraphLightningModel(model, optimizer=optimizer, score_dim=score_dim, print_log=print_log)
    else:
        return TorchLightningModule(model, optimizer=optimizer, score_dim=score_dim, print_log=print_log)


def create_trainer(model: nn.Module, training_info: dict, accelerator: str) -> L.Trainer:
    """Create lightning trainer."""

    devices = 1
    precision = '16-mixed' if torch.cuda.is_available() and accelerator == 'gpu' else '32'

    loggers = csv_logger(training_info)

    return L.Trainer(
        logger=loggers,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=config['Train']['max_epochs'],
        log_every_n_steps=config['Train']['log_every_n_steps'],
        num_sanity_val_steps=config['Train']['num_sanity_val_steps'],
        benchmark=True,
        deterministic=False,
        callbacks=[ModelCheckpoint(
            monitor=config['Train']['ckpt_monitor'],
            mode=config['Train']['ckpt_mode'],
            save_top_k=config['Train']['ckpt_top_k'],
            save_last=True,
            filename='{epoch}-{valid_auc:.3f}-{valid_accuracy:.3f}',
        )],
    )


def train(model: nn.Module, model_description: str, model_hparams: dict,
          accelerator: str, lr: float, graph: bool, pi_scale: bool = False):
    """Main training function."""

    L.seed_everything(random_seed)

    data_module = create_data_module(graph=graph, pi_scale=pi_scale)
    lightning_model = create_lightning_model(model=model, graph=graph, lr=lr)
    training_info = create_training_info(model, model_description, model_hparams, lr=lr)
    trainer = create_trainer(model, training_info, accelerator)

    # Training and testing
    if eval(config['Settings']['mode'][0]):
        if eval(config['Settings']['mode'][1]):
            trainer.fit(lightning_model, datamodule=data_module)
        else:
            trainer.fit(lightning_model, train_dataloaders=data_module.train_dataloader())

    if eval(config['Settings']['mode'][2]):
        trainer.test(lightning_model, datamodule=data_module, ckpt_path='best')

    return training_info['name']


# Training functions for each model type
def train_quantum(model_class: nn.Module, model_hparams: dict, model_name: str, pi_scale: bool, lr: float):
    """Train quantum model."""

    print(f"\n{'='*80}")
    print(f"🔬 Training {model_name}")
    print(f"{'='*80}")

    start_time = time.time()

    model = model_class(score_dim=score_dim, **model_hparams)

    num_ir_qubits = model_hparams['num_ir_qubits']
    num_nr_qubits = model_hparams['num_nr_qubits']
    num_layers = model_hparams['num_layers']
    num_reupload = model_hparams['num_reupload']
    dropout = model_hparams['dropout']
    model_description = f"nI{num_ir_qubits}_nQ{num_nr_qubits}_L{num_layers}_R{num_reupload}_D{dropout:.2f}"

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    name = train(model, model_description, model_hparams, accelerator, lr=lr, graph=False, pi_scale=pi_scale)

    elapsed = time.time() - start_time
    print(f"\n✅ {model_name} completed in {elapsed/60:.1f} minutes")

    return name


def train_mpgnn(model_hparams: dict, lr: float):
    """Train MPGNN model."""

    print(f"\n{'='*80}")
    print(f"🔬 Training MPGNN (Classical Message Passing GNN)")
    print(f"{'='*80}")

    start_time = time.time()

    model = ClassicalMPGNN(score_dim=score_dim, **model_hparams)

    phi_out = model_hparams['phi_out']
    phi_hidden = model_hparams['phi_hidden']
    phi_layers = model_hparams['phi_layers']
    dropout = model_hparams['dropout']
    model_description = f"O{phi_out}_H{phi_hidden}_L{phi_layers}_D{dropout:.2f}"

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    name = train(model, model_description, model_hparams, accelerator, lr=lr, graph=True)

    elapsed = time.time() - start_time
    print(f"\n✅ MPGNN completed in {elapsed/60:.1f} minutes")

    return name


def train_benchmark(model_class: nn.Module, model_name: str, lr: float):
    """Train benchmark models (PFN, ParT, PNet)."""

    print(f"\n{'='*80}")
    print(f"🔬 Training {model_name}")
    print(f"{'='*80}")

    start_time = time.time()

    with open('configs/benchmark.yaml', 'r') as file:
        hparams = yaml.safe_load(file)[model_class.__name__]
        model_description = ''

    model = model_class(score_dim=score_dim, parameters=hparams)

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    if model_class == ParticleFlowNetwork:
        name = train(model, model_description, hparams, accelerator, lr=lr, graph=True)
    else:
        name = train(model, model_description, hparams, accelerator, lr=lr, graph=False)

    elapsed = time.time() - start_time
    print(f"\n✅ {model_name} completed in {elapsed/60:.1f} minutes")

    return name


# Main training sequence
if __name__ == "__main__":
    print("\n" + "="*80)
    print("Starting Training Sequence - 6 Models Total")
    print("="*80)

    overall_start = time.time()
    results = {}

    # 1. QCGNN with 3 qubits
    qcgnn_3_hparams = {
        'num_ir_qubits': 4,
        'num_nr_qubits': 3,
        'num_layers': 1,
        'num_reupload': 2,
        'dropout': 0.0,
        'vqc_ansatz': qml.StronglyEntanglingLayers
    }
    results['QCGNN-3'] = train_quantum(
        model_class=QuantumRotQCGNN,
        model_hparams=qcgnn_3_hparams,
        model_name="QCGNN (3 qubits)",
        pi_scale=True,
        lr=1e-3
    )

    # 2. QCGNN with 6 qubits
    qcgnn_6_hparams = {
        'num_ir_qubits': 4,
        'num_nr_qubits': 6,
        'num_layers': 2,
        'num_reupload': 2,
        'dropout': 0.0,
        'vqc_ansatz': qml.StronglyEntanglingLayers
    }
    results['QCGNN-6'] = train_quantum(
        model_class=QuantumRotQCGNN,
        model_hparams=qcgnn_6_hparams,
        model_name="QCGNN (6 qubits)",
        pi_scale=True,
        lr=1e-3
    )

    # 3. MPGNN (Classical baseline)
    mpgnn_hparams = {
        'phi_in': 6,
        'phi_out': 64,
        'phi_layers': 2,
        'phi_hidden': 64,
        'mlp_hidden': 64,
        'dropout': 0.0
    }
    results['MPGNN'] = train_mpgnn(model_hparams=mpgnn_hparams, lr=1e-3)

    # 4. Particle Flow Network
    results['PFN'] = train_benchmark(
        model_class=ParticleFlowNetwork,
        model_name="Particle Flow Network (PFN)",
        lr=1e-3
    )

    # 5. Particle Transformer
    results['ParT'] = train_benchmark(
        model_class=ParticleTransformer,
        model_name="Particle Transformer (ParT)",
        lr=1e-3
    )

    # 6. Particle Net
    results['PNet'] = train_benchmark(
        model_class=ParticleNet,
        model_name="Particle Network (PNet)",
        lr=1e-3
    )

    # Summary
    total_time = time.time() - overall_start
    print("\n" + "="*80)
    print("🎉 ALL TRAINING COMPLETED!")
    print("="*80)
    print(f"\nTotal training time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print("\nModels trained:")
    for i, (model_name, result_name) in enumerate(results.items(), 1):
        print(f"  {i}. {model_name}: {result_name}")

    print(f"\n📁 Results saved to: training_logs/CSVLogger/")
    print("\nNext steps:")
    print("  - View results: Open training_result.ipynb")
    print("  - Check logs: Look in training_logs/CSVLogger/")
    print("  - Compare models: Use notebooks/analysis.ipynb")
