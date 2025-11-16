"""
Utility functions for Drug-Patient QGNN Pipeline.

This module provides:
- Seed setting for reproducibility
- Logging utilities
- Metric calculation helpers
- Visualization utilities
"""

import torch
import numpy as np
import random
from typing import Dict, List, Optional
import json


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends, 'mps'):
        # MPS backend doesn't need special seeding
        pass


def print_model_summary(model, drug_dim: int, patient_dim: int):
    """Print a summary of the model architecture.

    Args:
        model: PyTorch model
        drug_dim: Drug feature dimension
        patient_dim: Patient feature dimension
    """
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)

    # Model info
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        for key, value in info.items():
            print(f"{key:20s}: {value}")
    else:
        print(f"{'Model type':20s}: {type(model).__name__}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{'Total parameters':20s}: {total_params:,}")
    print(f"{'Trainable params':20s}: {trainable_params:,}")

    # Input/output shapes
    print(f"{'Input (drug)':20s}: ({drug_dim},)")
    print(f"{'Input (patient)':20s}: ({patient_dim},)")
    print(f"{'Output':20s}: (1,) [probability]")

    print("=" * 60 + "\n")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate classification metrics.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probabilities (optional, for AUC)

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix
    )

    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # AUC-ROC (requires probabilities)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['auc_roc'] = float('nan')

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negative'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negative'] = int(fn)
    metrics['true_positive'] = int(tp)

    # Specificity and sensitivity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics dictionary.

    Args:
        metrics: Dictionary of metric name -> value
        title: Title for the metrics display
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"{key:20s}: {value}")
        elif isinstance(value, (float, np.floating)):
            if np.isnan(value):
                print(f"{key:20s}: N/A")
            else:
                print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")

    print("=" * 60 + "\n")


def save_training_history(history: Dict[str, List[float]], filepath: str):
    """Save training history to JSON file.

    Args:
        history: Dictionary of metric name -> list of values
        filepath: Path to save JSON file
    """
    # Convert numpy types to Python types
    serializable_history = {}
    for key, values in history.items():
        serializable_history[key] = [
            float(v) if not np.isnan(v) else None
            for v in values
        ]

    with open(filepath, 'w') as f:
        json.dump(serializable_history, f, indent=2)

    print(f"Training history saved to {filepath}")


def load_training_history(filepath: str) -> Dict[str, List[float]]:
    """Load training history from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary of metric name -> list of values
    """
    with open(filepath, 'r') as f:
        history = json.load(f)

    # Convert None back to NaN
    for key, values in history.items():
        history[key] = [
            float(v) if v is not None else float('nan')
            for v in values
        ]

    print(f"Training history loaded from {filepath}")
    return history


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """Plot training history curves.

    Args:
        history: Training history dictionary
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available. Cannot plot training history.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss plot
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o', markersize=3)
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC plot
    if 'val_auc' in history:
        # Filter out NaN values
        auc_values = [v for v in history['val_auc'] if not np.isnan(v)]
        if len(auc_values) > 0:
            epochs = [i for i, v in enumerate(history['val_auc']) if not np.isnan(v)]
            axes[2].plot(epochs, auc_values, label='Val AUC-ROC', marker='s', markersize=3, color='green')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('AUC-ROC')
            axes[2].set_title('Validation AUC-ROC')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def get_device_info() -> Dict[str, any]:
    """Get information about available compute devices.

    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }

    if info['cuda_available']:
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0)
        info['cuda_memory_reserved'] = torch.cuda.memory_reserved(0)

    return info


def print_device_info():
    """Print information about available compute devices."""
    info = get_device_info()

    print("\n" + "=" * 60)
    print("Device Information")
    print("=" * 60)

    print(f"{'CUDA Available':20s}: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"{'CUDA Devices':20s}: {info['cuda_device_count']}")
        print(f"{'CUDA Device Name':20s}: {info['cuda_device_name']}")

    print(f"{'MPS Available':20s}: {info['mps_available']}")

    if not info['cuda_available'] and not info['mps_available']:
        print("\nNo GPU acceleration available. Training will use CPU.")

    print("=" * 60 + "\n")


def estimate_training_time(
    n_samples: int,
    batch_size: int,
    epochs: int,
    time_per_batch: float = 0.1
) -> str:
    """Estimate total training time.

    Args:
        n_samples: Number of training samples
        batch_size: Batch size
        epochs: Number of epochs
        time_per_batch: Estimated time per batch in seconds

    Returns:
        Formatted time estimate string
    """
    n_batches = (n_samples + batch_size - 1) // batch_size
    total_batches = n_batches * epochs
    total_seconds = total_batches * time_per_batch

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"
