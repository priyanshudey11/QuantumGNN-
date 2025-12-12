"""Evaluation and metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    brier_score_loss, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
from typing import Dict, Tuple, List
import json


def compute_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUROC."""
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return 0.0


def compute_auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUPRC (average precision)."""
    try:
        return average_precision_score(y_true, y_pred)
    except:
        return 0.0


def compute_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Brier score."""
    try:
        return brier_score_loss(y_true, y_pred)
    except:
        return 0.0


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ece = 0.0
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i+1])
        if in_bin.sum() > 0:
            bin_acc = y_true[in_bin].mean()
            bin_conf = y_pred[in_bin].mean()
            bin_counts.append(in_bin.sum())
            ece += in_bin.sum() * abs(bin_acc - bin_conf)
    
    ece /= len(y_true) if len(y_true) > 0 else 1
    return ece


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute all metrics."""
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    metrics = {
        'auroc': compute_auroc(y_true, y_pred),
        'auprc': compute_auprc(y_true, y_pred),
        'brier': compute_brier(y_true, y_pred),
        'ece': compute_ece(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
    }
    
    return metrics


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 42
) -> Dict:
    """Compute metrics with bootstrap confidence intervals."""
    rng = np.random.RandomState(seed)
    
    metrics_list = []
    
    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        if len(np.unique(y_true_boot)) > 1:  # Need both classes
            metrics = compute_metrics(y_true_boot, y_pred_boot)
            metrics_list.append(metrics)
    
    if not metrics_list:
        return {}
    
    # Aggregate
    result = {}
    for key in metrics_list[0].keys():
        vals = [m[key] for m in metrics_list if not np.isnan(m.get(key, np.nan))]
        if vals:
            result[key] = {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'ci95': (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
            }
    
    return result


class MetricsReporter:
    """Generate evaluation reports."""
    
    @staticmethod
    def generate_summary(
        test_preds: np.ndarray,
        test_labels: np.ndarray,
        binding_sites_by_pred: Dict = None,
        output_path: str = None
    ) -> str:
        """Generate summary markdown report."""
        
        # Overall metrics
        overall_metrics = compute_metrics(test_labels, test_preds)
        
        report = "# Drug-Protein Interaction Evaluation Report\n\n"
        
        report += "## Overall Metrics\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        for key, val in overall_metrics.items():
            report += f"| {key} | {val:.4f} |\n"
        
        # Metrics by binding site (if available)
        if binding_sites_by_pred:
            report += "\n## Metrics by Binding Site\n\n"
            
            for binding_site, indices in binding_sites_by_pred.items():
                if len(indices) > 0:
                    site_labels = test_labels[indices]
                    site_preds = test_preds[indices]
                    site_metrics = compute_metrics(site_labels, site_preds)
                    
                    report += f"\n### {binding_site.upper()} (n={len(indices)})\n\n"
                    report += "| Metric | Value |\n"
                    report += "|--------|-------|\n"
                    for key, val in site_metrics.items():
                        report += f"| {key} | {val:.4f} |\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report


def generate_roc_data(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Generate ROC curve data."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': float(roc_auc)
    }


def generate_pr_data(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Generate PR curve data."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    
    return {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'ap': float(ap)
    }
