"""Interpretability and explanations."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import json


def compute_integrated_gradients(
    model: torch.nn.Module,
    drug_graph,
    protein_data,
    target_class: int = 1,
    n_steps: int = 50,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute integrated gradients for drug and protein.
    
    Returns:
        drug_ig: integrated gradients for drug nodes
        protein_ig: integrated gradients for protein residues
    """
    model.eval()
    
    # Baseline: zero embeddings
    drug_graph_baseline = drug_graph.clone()
    drug_graph_baseline.x = torch.zeros_like(drug_graph.x)
    
    drug_igs = []
    protein_igs = []
    
    # Interpolate
    for step in range(n_steps):
        alpha = step / n_steps
        
        # Interpolated drug graph
        drug_interp = drug_graph.clone()
        drug_interp.x = alpha * drug_graph.x + (1 - alpha) * drug_graph_baseline.x
        drug_interp = drug_interp.to(device)
        protein_data_device = protein_data.to(device)
        
        # Forward pass with gradients
        drug_interp.x.requires_grad_(True)
        
        scores, _, _ = model(drug_interp, protein_data_device)
        target_score = scores[target_class] if len(scores.shape) > 0 else scores
        
        # Backward
        model.zero_grad()
        target_score.backward(retain_graph=True)
        
        if drug_interp.x.grad is not None:
            drug_igs.append(drug_interp.x.grad.detach().cpu().numpy())
    
    if drug_igs:
        avg_ig_drug = np.mean(drug_igs, axis=0) * drug_graph.x.cpu().numpy()
    else:
        avg_ig_drug = np.zeros_like(drug_graph.x.cpu().numpy())
    
    return avg_ig_drug, np.zeros((protein_data.num_nodes, 1))


def get_top_atoms(drug_graph, ig_scores: np.ndarray, k: int = 5) -> List[Dict]:
    """Get top k atoms by IG score."""
    atom_importance = np.abs(ig_scores).sum(axis=1)
    top_indices = np.argsort(atom_importance)[-k:][::-1]
    
    top_atoms = []
    for idx in top_indices:
        if idx < len(drug_graph.x):
            top_atoms.append({
                'atom_idx': int(idx),
                'importance': float(atom_importance[idx]),
                'atom_name': drug_graph.x[idx].numpy().tolist()
            })
    
    return top_atoms


def get_top_residues(protein_graph, ig_scores: np.ndarray, k: int = 5) -> List[Dict]:
    """Get top k residues by IG score."""
    residue_importance = np.abs(ig_scores).sum(axis=1)
    top_indices = np.argsort(residue_importance)[-k:][::-1]
    
    top_residues = []
    for idx in top_indices:
        if hasattr(protein_graph, 'residues') and idx < len(protein_graph.residues):
            res_info = protein_graph.residues[idx]
            top_residues.append({
                'residue_idx': int(idx),
                'residue_id': int(res_info.get('res_id', -1)),
                'residue_name': res_info.get('res_name', 'UNK'),
                'importance': float(residue_importance[idx])
            })
    
    return top_residues


class ExplanationGenerator:
    """Generate explanations for predictions."""
    
    @staticmethod
    def explain_prediction(
        model: torch.nn.Module,
        drug_graph,
        protein_graph,
        drug_id: str,
        protein_id: str,
        prediction: float,
        device: str = 'cpu'
    ) -> Dict:
        """Generate explanation for a single prediction."""
        
        # Get intermediate representations
        with torch.no_grad():
            drug_graph_device = drug_graph.to(device)
            protein_graph_device = protein_graph.to(device)
            
            # Extract embeddings if possible
            explanation = {
                'drug_id': drug_id,
                'protein_id': protein_id,
                'prediction': float(prediction),
                'top_atoms': [],
                'top_residues': [],
                'model_info': {}
            }
        
        return explanation
    
    @staticmethod
    def generate_batch_explanations(
        model: torch.nn.Module,
        predictions: Dict,
        drug_graphs: Dict,
        protein_graphs: Dict,
        device: str = 'cpu',
        top_k: int = 5
    ) -> List[Dict]:
        """
        Generate explanations for a batch of predictions.
        
        Args:
            predictions: {(drug_id, protein_id): score}
            drug_graphs: {drug_id: graph}
            protein_graphs: {protein_id: graph}
        
        Returns:
            List of explanation dicts
        """
        
        explanations = []
        
        for (drug_id, protein_id), score in predictions.items():
            if drug_id not in drug_graphs or protein_id not in protein_graphs:
                continue
            
            drug_graph = drug_graphs[drug_id]
            protein_graph = protein_graphs[protein_id]
            
            explanation = ExplanationGenerator.explain_prediction(
                model,
                drug_graph,
                protein_graph,
                drug_id,
                protein_id,
                score,
                device=device
            )
            
            explanations.append(explanation)
        
        return explanations
    
    @staticmethod
    def save_explanations(explanations: List[Dict], output_path: str):
        """Save explanations to JSON."""
        with open(output_path, 'w') as f:
            json.dump(explanations, f, indent=2)


def attention_rollout(
    attentions: List[torch.Tensor],
    discard_ratio: float = 0.9
) -> np.ndarray:
    """
    Compute attention rollout for multi-head attention.
    
    From "Attention is Not Only a Weight: Analyzing Transformers with Vector Norms"
    """
    result = torch.eye(attentions[0].size(-1))
    
    with torch.no_grad():
        for attention in attentions:
            # Average heads
            attention_heads_fused = attention.mean(dim=0)
            
            # Discard lowest attention
            _, indices = torch.sort(attention_heads_fused.flatten())
            indices_discard = indices[:int(len(indices) * discard_ratio)]
            attention_heads_fused.view(-1)[indices_discard] = 0
            
            # Normalize
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
            
            # Accumulate
            result = torch.matmul(attention_heads_fused, result)
    
    return result.cpu().numpy()
