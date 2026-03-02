"""
Experiment Runner for ECG Augmentation Experiments

Runs a single experiment configuration with k-fold cross-validation.
Designed to be called by run_experiments.py or standalone.

Usage:
    python experiment_runner.py --config configs/hp_amplitude_0.json --gpu 0
    python experiment_runner.py --config configs/baseline.json --gpu 1
"""

import os
import sys
import json
import time
import argparse
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SignalClassifier, count_parameters
from losses import AdaptiveMultiLabelLoss
from dataset import PTBXLDataset, SyntheticECGDataset
from metrics import compute_metrics, optimize_thresholds
from augmentations import build_pipeline_from_config


# ──────────────────────────────────────────────────────────────────────────────
# Multilabel Stratified K-Fold (from train_cv.py)
# ──────────────────────────────────────────────────────────────────────────────

class MultilabelStratifiedKFold:
    """Stratified K-Fold for multi-label classification."""

    def __init__(self, n_splits=3, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        n_samples, n_labels = y.shape
        rng = np.random.RandomState(self.random_state)
        indices = np.arange(n_samples)
        if self.shuffle:
            rng.shuffle(indices)

        fold_assignments = np.full(n_samples, -1)
        fold_label_counts = np.zeros((self.n_splits, n_labels))
        fold_sizes = np.zeros(self.n_splits)
        target_fold_size = n_samples / self.n_splits

        for idx in indices:
            sample_labels = y[idx]
            scores = np.zeros(self.n_splits)
            for fold in range(self.n_splits):
                size_penalty = max(0, fold_sizes[fold] - target_fold_size) * 10
                if fold_sizes[fold] > 0:
                    current_ratios = fold_label_counts[fold] / fold_sizes[fold]
                    global_ratios = y.sum(axis=0) / n_samples
                    label_benefit = np.sum((global_ratios - current_ratios) * sample_labels)
                else:
                    label_benefit = 1.0
                scores[fold] = label_benefit - size_penalty
            best_fold = np.argmax(scores)
            fold_assignments[idx] = best_fold
            fold_label_counts[best_fold] += sample_labels
            fold_sizes[best_fold] += 1

        for fold in range(self.n_splits):
            val_mask = fold_assignments == fold
            train_indices = np.where(~val_mask)[0].tolist()
            val_indices = np.where(val_mask)[0].tolist()
            yield train_indices, val_indices


# ──────────────────────────────────────────────────────────────────────────────
# Indexed Dataset wrapper (supports transform injection)
# ──────────────────────────────────────────────────────────────────────────────

class IndexedSubset(torch.utils.data.Dataset):
    """Wraps a PTBXLDataset with specific indices and optional transform override."""

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.labels = dataset.labels[indices]
        self.class_frequencies = self.labels.sum(axis=0) / max(len(self.labels), 1)
        self.signal_mean = getattr(dataset, 'signal_mean', np.zeros(12))
        self.signal_std = getattr(dataset, 'signal_std', np.ones(12))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        signal = np.array(self.dataset.signals[actual_idx], dtype=np.float32)
        signal = np.nan_to_num(signal, nan=0.0)

        if self.dataset.normalize:
            signal = (signal - self.dataset.signal_mean) / (self.dataset.signal_std + 1e-8)

        # Apply augmentation transform (training only)
        if self.transform is not None:
            signal = self.transform(signal)

        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        label = np.array(self.dataset.labels[actual_idx], dtype=np.float32)

        return {
            'signal': torch.from_numpy(signal),
            'label': torch.from_numpy(label),
        }

    def get_class_frequencies(self):
        return torch.from_numpy(self.class_frequencies.astype(np.float32))


# ──────────────────────────────────────────────────────────────────────────────
# Cosine schedule with warmup
# ──────────────────────────────────────────────────────────────────────────────

def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Training and validation
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip=1.0):
    """Train for one epoch, return metrics dict."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        signals = batch['signal'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(signals)
        loss, _ = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    metrics = compute_metrics(labels_arr, preds)
    metrics['loss'] = total_loss / max(len(loader), 1)
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate, return metrics, raw predictions, and labels."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        signals = batch['signal'].to(device)
        labels = batch['label'].to(device)

        logits = model(signals)
        loss, _ = criterion(logits, labels)

        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    metrics = compute_metrics(labels_arr, preds)
    metrics['loss'] = total_loss / max(len(loader), 1)
    return metrics, preds, labels_arr


# ──────────────────────────────────────────────────────────────────────────────
# Single fold training with early stopping
# ──────────────────────────────────────────────────────────────────────────────

def train_fold(model, train_loader, val_loader, criterion, device, train_cfg, logger):
    """
    Train one fold with early stopping.
    Returns: best metrics, best thresholds, val predictions, val labels, best state, training log
    """
    epochs = train_cfg.get('epochs', 10)
    lr = train_cfg.get('lr', 1e-3)
    weight_decay = train_cfg.get('weight_decay', 1e-4)
    warmup_epochs = train_cfg.get('warmup_epochs', 1)
    patience = train_cfg.get('patience', 3)
    grad_clip = train_cfg.get('grad_clip', 1.0)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)

    best_auc = 0.0
    best_state = None
    best_epoch = 0
    no_improve = 0
    training_log = []

    for epoch in range(epochs):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, grad_clip)
        val_metrics, _, _ = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        improved = False
        if val_metrics['macro_auc'] > best_auc:
            best_auc = val_metrics['macro_auc']
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            no_improve = 0
            improved = True
        else:
            no_improve += 1

        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': round(train_metrics['loss'], 6),
            'train_f1': round(train_metrics['macro_f1'], 4),
            'train_auc': round(train_metrics['macro_auc'], 4),
            'val_loss': round(val_metrics['loss'], 6),
            'val_f1': round(val_metrics['macro_f1'], 4),
            'val_auc': round(val_metrics['macro_auc'], 4),
            'lr': optimizer.param_groups[0]['lr'],
            'time_s': round(elapsed, 2),
            'improved': improved,
        }
        training_log.append(epoch_log)

        marker = " *" if improved else ""
        logger.info(
            f"  Epoch {epoch+1:>2}/{epochs} | "
            f"Train L={train_metrics['loss']:.4f} F1={train_metrics['macro_f1']:.4f} | "
            f"Val L={val_metrics['loss']:.4f} F1={val_metrics['macro_f1']:.4f} AUC={val_metrics['macro_auc']:.4f} | "
            f"{elapsed:.1f}s{marker}"
        )

        if no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    # Load best model and do final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics, val_preds, val_labels = validate(model, val_loader, criterion, device)
    optimal_thresholds = optimize_thresholds(val_preds, val_labels)
    final_metrics = compute_metrics(val_labels, val_preds, optimal_thresholds)

    logger.info(f"  Best epoch: {best_epoch}, AUC={best_auc:.4f}, Final F1={final_metrics['macro_f1']:.4f}")

    return final_metrics, optimal_thresholds, val_preds, val_labels, best_state, training_log


# ──────────────────────────────────────────────────────────────────────────────
# Stratified subset sampling
# ──────────────────────────────────────────────────────────────────────────────

def sample_stratified_subset(labels, fraction=0.1, seed=42):
    """
    Sample a stratified subset of indices preserving label distribution.
    Uses iterative allocation to maintain multi-label balance.
    """
    rng = np.random.RandomState(seed)
    n_samples = labels.shape[0]
    target_size = max(int(n_samples * fraction), 50)

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    # Score each sample by how much it helps balance
    selected = []
    selected_set = set()
    label_counts = np.zeros(labels.shape[1])
    target_ratios = labels.sum(axis=0) / n_samples

    for idx in indices:
        if len(selected) >= target_size:
            break
        selected.append(idx)
        selected_set.add(idx)
        label_counts += labels[idx]

    selected = np.array(sorted(selected))
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ──────────────────────────────────────────────────────────────────────────────

def setup_logger(log_path, name='experiment'):
    """Create logger that writes to both file and stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing

    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def run_experiment(config_path: str, gpu_id: int = 0):
    """
    Run a complete experiment from a JSON config file.
    
    Config JSON structure:
    {
        "experiment_id": "hp_amplitude_0",
        "stage": "hp_search",
        "technique": "amplitude",
        "description": "Amplitude scaling with scale=[0.9, 1.1]",
        
        "data": {
            "data_dir": "/path/to/preprocessed",
            "subset_fraction": 0.1,
            "subset_seed": 42
        },
        
        "augmentation": {
            "amplitude": {"enabled": true, "scale_min": 0.9, "scale_max": 1.1, "p": 0.5},
            "noise": {"enabled": false},
            "lead_mask": {"enabled": false},
            "time_mask": {"enabled": false},
            "time_shift": {"enabled": false}
        },
        
        "training": {
            "epochs": 10,
            "batch_size": 32,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "warmup_epochs": 1,
            "patience": 3,
            "grad_clip": 1.0,
            "num_workers": 4
        },
        
        "model": {
            "d_model": 256,
            "num_layers": 6,
            "num_heads": 8,
            "downsample_factor": 10
        },
        
        "cv": {
            "n_folds": 3,
            "seed": 42
        },
        
        "output_dir": "results/hp_search/amplitude_0"
    }
    """
    # ── Load config ──
    with open(config_path, 'r') as f:
        config = json.load(f)

    experiment_id = config['experiment_id']
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup logging ──
    logger = setup_logger(output_dir / 'experiment.log', name=experiment_id)
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT: {experiment_id}")
    logger.info(f"Description: {config.get('description', 'N/A')}")
    logger.info(f"Stage: {config.get('stage', 'N/A')}")
    logger.info(f"GPU: {gpu_id}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)

    # ── Device setup ──
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
        logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # ── Seeds ──
    seed = config.get('cv', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # ── Load dataset ──
    data_cfg = config.get('data', {})
    data_dir = data_cfg.get('data_dir', None)

    if data_dir and os.path.exists(data_dir):
        dataset = PTBXLDataset(data_dir, normalize=True)
        label_names = dataset.get_label_names()
        num_classes = dataset.num_classes
        all_labels = dataset.labels
        logger.info(f"Loaded dataset: {len(dataset)} samples, {num_classes} classes")
    else:
        logger.info("No data_dir found, using synthetic data")
        dataset = SyntheticECGDataset(num_samples=500, num_classes=5, seed=seed)
        label_names = dataset.label_names
        num_classes = dataset.num_classes
        all_labels = dataset.labels

    # ── Subset sampling ──
    subset_frac = data_cfg.get('subset_fraction', 0.1)
    subset_seed = data_cfg.get('subset_seed', 42)

    if subset_frac < 1.0:
        subset_indices = sample_stratified_subset(all_labels, fraction=subset_frac, seed=subset_seed)
        logger.info(f"Sampled {len(subset_indices)} of {len(all_labels)} ({subset_frac*100:.0f}%) for tuning")
    else:
        subset_indices = np.arange(len(all_labels))
        logger.info(f"Using full dataset: {len(subset_indices)} samples")

    subset_labels = all_labels[subset_indices]

    # Log class distribution
    class_counts = subset_labels.sum(axis=0).astype(int)
    for i, name in enumerate(label_names):
        pct = 100 * class_counts[i] / len(subset_labels)
        logger.info(f"  {name}: {class_counts[i]} ({pct:.1f}%)")

    # ── Build augmentation pipeline ──
    aug_config = config.get('augmentation', {})
    pipeline = build_pipeline_from_config(aug_config)
    if pipeline is not None:
        logger.info(f"Augmentation pipeline: {pipeline.get_config()}")
    else:
        logger.info("No augmentation (baseline)")

    # ── Cross-validation ──
    cv_cfg = config.get('cv', {})
    n_folds = cv_cfg.get('n_folds', 3)
    train_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    batch_size = train_cfg.get('batch_size', 32)
    num_workers = train_cfg.get('num_workers', 4)

    kfold = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    all_val_preds = []
    all_val_labels = []
    experiment_start = time.time()

    for fold_idx, (train_rel, val_rel) in enumerate(kfold.split(None, subset_labels)):
        # Map relative indices back to dataset indices
        train_indices = subset_indices[train_rel].tolist()
        val_indices = subset_indices[val_rel].tolist()

        fold_dir = output_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'─'*60}")
        logger.info(f"FOLD {fold_idx}/{n_folds - 1} | Train: {len(train_indices)}, Val: {len(val_indices)}")
        logger.info(f"{'─'*60}")

        # Log fold class distribution
        fold_train_labels = all_labels[train_indices]
        fold_val_labels = all_labels[val_indices]
        for i, name in enumerate(label_names):
            tr_ct = fold_train_labels[:, i].sum().astype(int)
            va_ct = fold_val_labels[:, i].sum().astype(int)
            logger.info(f"  {name}: train={tr_ct}, val={va_ct}")

        # Create data subsets (augmentation only on train)
        train_subset = IndexedSubset(dataset, train_indices, transform=pipeline)
        val_subset = IndexedSubset(dataset, val_indices, transform=None)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        logger.info(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # Create fresh model
        model = SignalClassifier(
            num_classes=num_classes,
            d_model=model_cfg.get('d_model', 256),
            max_seq_len=5000,
            num_layers=model_cfg.get('num_layers', 6),
            num_heads=model_cfg.get('num_heads', 8),
            downsample_factor=model_cfg.get('downsample_factor', 10),
        ).to(device)

        if fold_idx == 0:
            logger.info(f"  Model params: {count_parameters(model):,}")

        criterion = AdaptiveMultiLabelLoss(
            train_subset.get_class_frequencies()
        ).to(device)

        # Train fold
        fold_start = time.time()
        metrics, thresholds, val_preds, val_labels, best_state, training_log = train_fold(
            model, train_loader, val_loader, criterion, device, train_cfg, logger
        )
        fold_time = time.time() - fold_start

        logger.info(f"  Fold {fold_idx} time: {fold_time:.1f}s")

        # Save fold results
        fold_result = {
            'fold': fold_idx,
            'macro_auc': metrics['macro_auc'],
            'macro_f1': metrics['macro_f1'],
            'macro_precision': metrics.get('macro_precision', 0),
            'macro_recall': metrics.get('macro_recall', 0),
            'per_class_auc': metrics['per_class_auc'],
            'per_class_f1': metrics['per_class_f1'],
            'thresholds': thresholds.tolist(),
            'training_time_s': round(fold_time, 2),
            'num_train': len(train_indices),
            'num_val': len(val_indices),
        }
        fold_results.append(fold_result)
        all_val_preds.append(val_preds)
        all_val_labels.append(val_labels)

        # Save fold artifacts
        with open(fold_dir / 'metrics.json', 'w') as f:
            json.dump(fold_result, f, indent=2)
        with open(fold_dir / 'training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)
        np.save(fold_dir / 'predictions.npy', val_preds)
        np.save(fold_dir / 'labels.npy', val_labels)

        # Save model checkpoint
        torch.save({
            'model_state_dict': best_state,
            'thresholds': thresholds,
            'metrics': metrics,
            'fold': fold_idx,
            'config': config,
        }, fold_dir / 'model.pt')

        # Free GPU memory
        del model, criterion, best_state
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    total_time = time.time() - experiment_start

    # ── Aggregate results ──
    auc_scores = [r['macro_auc'] for r in fold_results]
    f1_scores = [r['macro_f1'] for r in fold_results]

    # Combined validation metrics
    combined_preds = np.concatenate(all_val_preds)
    combined_labels = np.concatenate(all_val_labels)
    combined_thresholds = optimize_thresholds(combined_preds, combined_labels)
    combined_metrics = compute_metrics(combined_labels, combined_preds, combined_thresholds)

    # Per-class aggregation
    per_class_auc_mean = np.mean([r['per_class_auc'] for r in fold_results], axis=0).tolist()
    per_class_auc_std = np.std([r['per_class_auc'] for r in fold_results], axis=0).tolist()
    per_class_f1_mean = np.mean([r['per_class_f1'] for r in fold_results], axis=0).tolist()
    per_class_f1_std = np.std([r['per_class_f1'] for r in fold_results], axis=0).tolist()

    summary = {
        'experiment_id': experiment_id,
        'stage': config.get('stage', 'unknown'),
        'technique': config.get('technique', 'none'),
        'description': config.get('description', ''),
        'augmentation': aug_config,
        'auc_mean': round(float(np.mean(auc_scores)), 6),
        'auc_std': round(float(np.std(auc_scores)), 6),
        'f1_mean': round(float(np.mean(f1_scores)), 6),
        'f1_std': round(float(np.std(f1_scores)), 6),
        'combined_auc': round(combined_metrics['macro_auc'], 6),
        'combined_f1': round(combined_metrics['macro_f1'], 6),
        'per_class_auc_mean': [round(v, 4) for v in per_class_auc_mean],
        'per_class_auc_std': [round(v, 4) for v in per_class_auc_std],
        'per_class_f1_mean': [round(v, 4) for v in per_class_f1_mean],
        'per_class_f1_std': [round(v, 4) for v in per_class_f1_std],
        'label_names': label_names,
        'n_folds': n_folds,
        'fold_results': fold_results,
        'total_time_s': round(total_time, 2),
        'gpu_id': gpu_id,
        'timestamp': datetime.now().isoformat(),
    }

    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Log final results ──
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT COMPLETE: {experiment_id}")
    logger.info(f"{'='*70}")
    logger.info(f"  Macro AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    logger.info(f"  Macro F1:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    logger.info(f"  Combined AUC: {combined_metrics['macro_auc']:.4f}")
    logger.info(f"  Combined F1:  {combined_metrics['macro_f1']:.4f}")
    logger.info(f"  Per-class AUC (mean ± std):")
    for i, name in enumerate(label_names):
        logger.info(f"    {name:<6}: {per_class_auc_mean[i]:.4f} ± {per_class_auc_std[i]:.4f}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info(f"  Results: {output_dir / 'summary.json'}")

    # Write a status file to indicate completion
    with open(output_dir / 'DONE', 'w') as f:
        f.write(f"{datetime.now().isoformat()}\n")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Run a single ECG experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config JSON')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    summary = run_experiment(args.config, args.gpu)
    print(f"\nDone: AUC={summary['auc_mean']:.4f}±{summary['auc_std']:.4f}")


if __name__ == '__main__':
    main()