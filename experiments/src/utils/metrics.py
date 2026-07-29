"""
Evaluation metrics for CodeFault.
"""

from typing import Dict, List
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)


def compute_metrics(labels: List[float], probs: List[float], threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        labels: Ground truth labels (0 or 1)
        probs: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    labels = np.array(labels)
    probs = np.array(probs)
    preds = (probs >= threshold).astype(int)
    
    metrics = {}
    
    # AUC
    try:
        metrics['auc'] = roc_auc_score(labels, probs)
    except ValueError:
        metrics['auc'] = 0.0
    
    # F1, Precision, Recall
    metrics['f1'] = f1_score(labels, preds, zero_division=0)
    metrics['precision'] = precision_score(labels, preds, zero_division=0)
    metrics['recall'] = recall_score(labels, preds, zero_division=0)
    
    # Average Precision (AP)
    try:
        metrics['ap'] = average_precision_score(labels, probs)
    except ValueError:
        metrics['ap'] = 0.0
    
    # Precision@K (top 20%)
    metrics['p_at_20'] = precision_at_k(labels, probs, k=0.2)
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics['tp'] = int(tp)
    metrics['fp'] = int(fp)
    metrics['tn'] = int(tn)
    metrics['fn'] = int(fn)
    
    return metrics


def precision_at_k(labels: np.ndarray, probs: np.ndarray, k: float = 0.2) -> float:
    """
    Compute Precision@K where K is a percentage of samples.
    
    Args:
        labels: Ground truth labels
        probs: Predicted probabilities
        k: Percentage of top predictions to consider
        
    Returns:
        Precision at K
    """
    n = len(labels)
    top_k = int(n * k)
    
    if top_k == 0:
        return 0.0
    
    # Get indices of top-k predictions
    top_indices = np.argsort(probs)[-top_k:]
    
    # Compute precision
    precision = labels[top_indices].sum() / top_k
    
    return float(precision)


def effort_at_recall(labels: np.ndarray, probs: np.ndarray, target_recall: float = 0.8) -> float:
    """
    Compute effort (percentage of samples to inspect) to achieve target recall.
    
    Args:
        labels: Ground truth labels
        probs: Predicted probabilities
        target_recall: Target recall to achieve
        
    Returns:
        Percentage of samples that need to be inspected
    """
    n = len(labels)
    total_positives = labels.sum()
    
    if total_positives == 0:
        return 1.0
    
    # Sort by probability descending
    sorted_indices = np.argsort(probs)[::-1]
    
    cumsum_positives = 0
    for i, idx in enumerate(sorted_indices):
        cumsum_positives += labels[idx]
        current_recall = cumsum_positives / total_positives
        
        if current_recall >= target_recall:
            return (i + 1) / n
    
    return 1.0


def compute_cross_project_metrics(results: Dict[str, Dict]) -> Dict[str, float]:
    """
    Aggregate metrics across projects.
    
    Args:
        results: Dictionary mapping project name to metrics
        
    Returns:
        Aggregated metrics
    """
    agg_metrics = {}
    
    metric_names = ['auc', 'f1', 'precision', 'recall', 'ap', 'p_at_20']
    
    for metric in metric_names:
        values = [r[metric] for r in results.values() if metric in r]
        if values:
            agg_metrics[f'avg_{metric}'] = np.mean(values)
            agg_metrics[f'std_{metric}'] = np.std(values)
    
    return agg_metrics


if __name__ == "__main__":
    # Test metrics
    labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    probs = [0.9, 0.3, 0.8, 0.7, 0.2, 0.6, 0.4, 0.1, 0.85, 0.75]
    
    metrics = compute_metrics(labels, probs)
    print("Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
