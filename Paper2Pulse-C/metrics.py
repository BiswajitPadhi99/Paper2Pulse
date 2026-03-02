"""
Evaluation Metrics for Multi-Label ECG Classification

Primary metrics: Macro F1, Macro AUC
Includes threshold optimization (Stage 2 of training).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_metrics(labels: np.ndarray, probs: np.ndarray,
                   thresholds: Optional[np.ndarray] = None) -> Dict:
    """
    Compute multi-label classification metrics.
    
    Args:
        labels: Ground truth labels [N, C]
        probs: Predicted probabilities [N, C]
        thresholds: Per-class thresholds (default 0.5)
    
    Returns:
        Dict with macro_f1, macro_auc, per_class metrics
    """
    num_classes = labels.shape[1]
    if thresholds is None:
        thresholds = np.ones(num_classes) * 0.5
    
    binary_preds = (probs > thresholds).astype(int)
    f1_scores = []
    auc_scores = []
    precisions = []
    recalls = []
    
    for c in range(num_classes):
        tp = ((binary_preds[:, c] == 1) & (labels[:, c] == 1)).sum()
        fp = ((binary_preds[:, c] == 1) & (labels[:, c] == 0)).sum()
        fn = ((binary_preds[:, c] == 0) & (labels[:, c] == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_scores.append(float(f1))
        auc_scores.append(float(_compute_auc(labels[:, c], probs[:, c])))
    
    return {
        'macro_f1': float(np.mean(f1_scores)),
        'macro_auc': float(np.mean(auc_scores)),
        'macro_precision': float(np.mean(precisions)),
        'macro_recall': float(np.mean(recalls)),
        'per_class_f1': f1_scores,
        'per_class_auc': auc_scores,
    }


def _compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute AUC (Area Under ROC Curve) without sklearn.
    
    Args:
        y_true: Binary labels
        y_score: Predicted scores/probabilities
    
    Returns:
        AUC score
    """
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Sort by descending score
    desc_idx = np.argsort(y_score)[::-1]
    y_sorted = y_true[desc_idx]
    
    # Compute TPR and FPR
    tpr = np.concatenate([[0], np.cumsum(y_sorted) / n_pos])
    fpr = np.concatenate([[0], np.cumsum(1 - y_sorted) / n_neg])
    
    # Compute AUC using trapezoidal rule
    return float(np.trapz(tpr, fpr))


def optimize_thresholds(probs: np.ndarray, labels: np.ndarray,
                        threshold_range: Tuple[float, float] = (0.1, 0.9),
                        num_steps: int = 81) -> np.ndarray:
    """
    Grid search for optimal per-class thresholds maximizing F1.
    
    Args:
        probs: Predicted probabilities [N, C]
        labels: Ground truth labels [N, C]
        threshold_range: (min, max) threshold values to search
        num_steps: Number of threshold values to try
    
    Returns:
        Optimal threshold for each class [C]
    """
    num_classes = probs.shape[1]
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_steps)
    optimal = np.zeros(num_classes)
    
    for c in range(num_classes):
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in thresholds:
            preds = (probs[:, c] > thresh).astype(int)
            tp = ((preds == 1) & (labels[:, c] == 1)).sum()
            fp = ((preds == 1) & (labels[:, c] == 0)).sum()
            fn = ((preds == 0) & (labels[:, c] == 1)).sum()
            
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        optimal[c] = best_thresh
    
    return optimal


def print_metrics(metrics: Dict, class_names: Optional[List[str]] = None) -> None:
    """
    Pretty print metrics.
    
    Args:
        metrics: Dict from compute_metrics()
        class_names: Optional list of class names
    """
    num_classes = len(metrics['per_class_f1'])
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    print("\n" + "=" * 50)
    print(f"{'Macro F1:':<20} {metrics['macro_f1']:.4f}")
    print(f"{'Macro AUC:':<20} {metrics['macro_auc']:.4f}")
    print("-" * 50)
    print(f"{'Class':<10} {'F1':>8} {'AUC':>8}")
    print("-" * 50)
    for i in range(num_classes):
        print(f"{class_names[i]:<10} {metrics['per_class_f1'][i]:>8.4f} {metrics['per_class_auc'][i]:>8.4f}")
    print("=" * 50)


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    labels = np.random.randint(0, 2, (100, 5)).astype(float)
    probs = np.random.rand(100, 5)
    
    metrics = compute_metrics(labels, probs)
    print("Test compute_metrics:")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
    
    thresholds = optimize_thresholds(probs, labels)
    print(f"\nOptimal thresholds: {thresholds}")
    
    print_metrics(metrics, ['A', 'B', 'C', 'D', 'E'])
    print("\n✓ All functions working!")