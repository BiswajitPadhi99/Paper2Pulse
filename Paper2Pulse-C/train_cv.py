"""
Training Script with Stratified K-Fold Cross-Validation for PTB-XL

Features:
- Verbose logging with timing information
- Checkpoint saving every N epochs
- Per-class metrics tracking
- GPU memory monitoring
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from model import SignalClassifier, count_parameters
from losses import AdaptiveMultiLabelLoss
from dataset import PTBXLDataset, SyntheticECGDataset
from metrics import compute_metrics, optimize_thresholds, print_metrics


def print_header(text, char="=", width=80):
    """Print formatted header."""
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def print_subheader(text, char="-", width=60):
    """Print formatted subheader."""
    print(f"\n{char * width}")
    print(f" {text}")
    print(char * width)


def format_time(seconds):
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_gpu_memory():
    """Get GPU memory usage if available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"{allocated:.2f}GB / {reserved:.2f}GB"
    return "N/A"


class MultilabelStratifiedKFold:
    """Stratified K-Fold for multi-label classification."""
    
    def __init__(self, n_splits: int = 10, shuffle: bool = True, random_state: int = 42):
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


def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, criterion, optimizer, scheduler, device, 
                epoch, total_epochs, grad_clip=1.0, verbose=True):
    """Train for one epoch with progress tracking."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    num_batches = len(loader)
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(loader):
        signals = batch['signal'].to(device)
        labels = batch['label'].to(device)
        
        # Debug first batch of first epoch
        if epoch == 0 and batch_idx == 0:
            print(f"\n  [DEBUG] First batch stats:")
            print(f"    Signal shape: {signals.shape}")
            print(f"    Signal range: [{signals.min().item():.4f}, {signals.max().item():.4f}]")
            print(f"    Signal mean: {signals.mean().item():.4f}")
            print(f"    Signal NaN count: {torch.isnan(signals).sum().item()}")
            print(f"    Signal Inf count: {torch.isinf(signals).sum().item()}")
            print(f"    Labels shape: {labels.shape}")
            print(f"    Labels sum: {labels.sum().item():.0f}")
        
        optimizer.zero_grad()
        logits = model(signals)
        
        # Debug first batch
        if epoch == 0 and batch_idx == 0:
            print(f"    Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            print(f"    Logits NaN count: {torch.isnan(logits).sum().item()}")
        
        loss, _ = criterion(logits, labels)
        
        # Debug first batch
        if epoch == 0 and batch_idx == 0:
            print(f"    Loss: {loss.item():.4f}")
            print()
        
        loss.backward()
        
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() if not np.isnan(loss.item()) else 0.0
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        # Progress update every 10% of batches
        if verbose and (batch_idx + 1) % max(1, num_batches // 10) == 0:
            progress = (batch_idx + 1) / num_batches * 100
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - epoch_start
            eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
            print(f"\r    Epoch {epoch+1}/{total_epochs} | "
                  f"Batch {batch_idx+1}/{num_batches} ({progress:.0f}%) | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"ETA: {format_time(eta)}", end="", flush=True)
    
    if verbose:
        print()  # New line after progress
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    metrics = compute_metrics(labels, preds)
    metrics['loss'] = total_loss / len(loader)
    metrics['time'] = time.time() - epoch_start
    
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device, thresholds=None):
    """Validate model."""
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
    labels = np.concatenate(all_labels)
    metrics = compute_metrics(labels, preds, thresholds)
    metrics['loss'] = total_loss / len(loader)
    return metrics, preds, labels


def save_checkpoint(model, optimizer, scheduler, epoch, fold_idx, metrics, 
                    save_dir, label_names, args):
    """Save training checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'checkpoint_fold{fold_idx+1}_epoch{epoch+1}.pt')
    
    torch.save({
        'epoch': epoch,
        'fold': fold_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'label_names': label_names,
        'args': vars(args)
    }, checkpoint_path)
    
    print(f"    💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def train_fold(model, train_loader, val_loader, criterion, device, args, 
               fold_idx, label_names):
    """Train a single fold with verbose output and checkpointing."""
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)
    
    best_f1 = 0
    best_state = None
    best_epoch = 0
    fold_start = time.time()
    
    print(f"\n  {'Epoch':<8} {'Train Loss':<12} {'Train F1':<10} {'Val Loss':<12} {'Val F1':<10} {'Val AUC':<10} {'Time':<10} {'GPU Mem':<15}")
    print("  " + "-" * 95)
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            epoch, args.epochs, verbose=(epoch == 0)  # Only verbose first epoch
        )
        
        # Validate
        val_metrics, _, _ = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        gpu_mem = get_gpu_memory()
        
        # Print epoch summary
        improved = ""
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            improved = " ⭐"
        
        print(f"  {epoch+1:<8} {train_metrics['loss']:<12.4f} {train_metrics['macro_f1']:<10.4f} "
              f"{val_metrics['loss']:<12.4f} {val_metrics['macro_f1']:<10.4f} {val_metrics['macro_auc']:<10.4f} "
              f"{format_time(epoch_time):<10} {gpu_mem:<15}{improved}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, fold_idx, 
                          val_metrics, args.save_dir, label_names, args)
    
    fold_time = time.time() - fold_start
    print(f"\n  Fold {fold_idx+1} completed in {format_time(fold_time)}")
    print(f"  Best validation F1: {best_f1:.4f} at epoch {best_epoch}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    # Final evaluation with threshold optimization
    _, val_preds, val_labels = validate(model, val_loader, criterion, device)
    optimal_thresholds = optimize_thresholds(val_preds, val_labels)
    final_metrics = compute_metrics(val_labels, val_preds, optimal_thresholds)
    
    # Print per-class metrics
    print(f"\n  Per-class F1 scores (with optimized thresholds):")
    for i, name in enumerate(label_names):
        thresh = optimal_thresholds[i]
        f1 = final_metrics['per_class_f1'][i]
        print(f"    {name:<10}: F1={f1:.4f}, threshold={thresh:.3f}")
    
    return final_metrics, optimal_thresholds, val_preds, val_labels, best_state


class IndexedDataset(torch.utils.data.Dataset):
    """Wrapper to access dataset by indices."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.labels = dataset.labels[indices]
        self.class_frequencies = self.labels.sum(axis=0) / len(self.labels)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def get_class_frequencies(self):
        return torch.from_numpy(self.class_frequencies).float()


def train_cv(args):
    """Main cross-validation training function."""
    
    # Print run configuration
    print_header(f"ECG CLASSIFIER - {args.n_folds}-FOLD CROSS-VALIDATION")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load dataset
    print_subheader("Loading Dataset")
    
    if args.data_dir:
        dataset = PTBXLDataset(args.data_dir, normalize=True)
        label_names = dataset.get_label_names()
        num_classes = dataset.num_classes
        labels_array = dataset.labels
        print(f"  Data directory: {args.data_dir}")
    else:
        print("  No data_dir specified, using synthetic data...")
        dataset = SyntheticECGDataset(
            num_samples=args.num_samples,
            num_classes=5,
            seq_length=5000,
            seed=args.seed
        )
        label_names = dataset.label_names
        num_classes = dataset.num_classes
        labels_array = dataset.labels
    
    print(f"  Total samples: {len(dataset)}")
    print(f"  Num classes: {num_classes}")
    print(f"  Labels: {label_names}")
    print(f"  Signal shape: [5000, 12]")
    
    # Print class distribution
    print(f"\n  Class distribution:")
    class_counts = labels_array.sum(axis=0).astype(int)
    for i, name in enumerate(label_names):
        pct = 100 * class_counts[i] / len(dataset)
        print(f"    {name:<10}: {class_counts[i]:>6} ({pct:>5.1f}%)")
    
    # Print configuration
    print_subheader("Training Configuration")
    print(f"  Folds: {args.n_folds}")
    print(f"  Epochs per fold: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Model dim: {args.d_model}")
    print(f"  Transformer layers: {args.num_layers}")
    print(f"  Attention heads: {args.num_heads}")
    print(f"  Downsample factor: {args.downsample_factor}x (5000 -> {5000//args.downsample_factor} timesteps)")
    print(f"  Checkpoint every: {args.save_every} epochs")
    print(f"  Save directory: {args.save_dir}")
    
    # Create folds
    print_subheader("Creating Stratified Folds")
    kfold = MultilabelStratifiedKFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=args.seed
    )
    
    # Storage for results
    fold_metrics = []
    fold_models = []
    all_val_preds = []
    all_val_labels = []
    all_val_indices = []
    
    cv_start_time = time.time()
    
    # Train each fold
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(None, labels_array)):
        print_header(f"FOLD {fold_idx + 1}/{args.n_folds}")
        
        print(f"  Train samples: {len(train_indices)}")
        print(f"  Val samples: {len(val_indices)}")
        
        # Print fold class distribution
        train_labels = labels_array[train_indices].sum(axis=0).astype(int)
        val_labels_dist = labels_array[val_indices].sum(axis=0).astype(int)
        print(f"\n  {'Class':<10} {'Train':>8} {'Val':>8}")
        print("  " + "-" * 28)
        for i, name in enumerate(label_names):
            print(f"  {name:<10} {train_labels[i]:>8} {val_labels_dist[i]:>8}")
        
        # Create data loaders
        train_subset = IndexedDataset(dataset, train_indices)
        val_subset = IndexedDataset(dataset, val_indices)
        
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size,
            shuffle=True, drop_last=True,
            num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            pin_memory=True
        )
        
        print(f"\n  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Create fresh model
        model = SignalClassifier(
            num_classes=num_classes,
            d_model=args.d_model,
            max_seq_len=5000,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            downsample_factor=args.downsample_factor,
        ).to(device)
        
        if fold_idx == 0:
            print(f"\n  Model parameters: {count_parameters(model):,}")
        
        # Create loss
        criterion = AdaptiveMultiLabelLoss(
            train_subset.get_class_frequencies()
        ).to(device)
        
        # Train fold
        print_subheader(f"Training Fold {fold_idx + 1}")
        
        metrics, thresholds, val_preds, val_labels, best_state = train_fold(
            model, train_loader, val_loader, criterion, device, args,
            fold_idx, label_names
        )
        
        fold_metrics.append(metrics)
        fold_models.append(best_state)
        all_val_preds.append(val_preds)
        all_val_labels.append(val_labels)
        all_val_indices.append(val_indices)
        
        print(f"\n  ✅ Fold {fold_idx + 1} Final: F1={metrics['macro_f1']:.4f}, AUC={metrics['macro_auc']:.4f}")
        
        # Save fold model
        fold_model_path = os.path.join(args.save_dir, f'best_model_fold{fold_idx+1}.pt')
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': best_state,
            'thresholds': thresholds,
            'metrics': metrics,
            'label_names': label_names,
            'fold': fold_idx,
            'args': vars(args)
        }, fold_model_path)
        print(f"  💾 Saved best model: {fold_model_path}")
    
    cv_time = time.time() - cv_start_time
    
    # Final Results
    print_header("CROSS-VALIDATION RESULTS")
    
    print(f"\n  Total training time: {format_time(cv_time)}")
    print(f"  Average time per fold: {format_time(cv_time / args.n_folds)}")
    
    # Per-fold summary
    print(f"\n  {'Fold':<8} {'Macro F1':>12} {'Macro AUC':>12}")
    print("  " + "-" * 34)
    for i, m in enumerate(fold_metrics):
        print(f"  {i+1:<8} {m['macro_f1']:>12.4f} {m['macro_auc']:>12.4f}")
    
    f1_scores = [m['macro_f1'] for m in fold_metrics]
    auc_scores = [m['macro_auc'] for m in fold_metrics]
    
    print("  " + "-" * 34)
    print(f"  {'Mean':<8} {np.mean(f1_scores):>12.4f} {np.mean(auc_scores):>12.4f}")
    print(f"  {'Std':<8} {np.std(f1_scores):>12.4f} {np.std(auc_scores):>12.4f}")
    print(f"  {'Min':<8} {np.min(f1_scores):>12.4f} {np.min(auc_scores):>12.4f}")
    print(f"  {'Max':<8} {np.max(f1_scores):>12.4f} {np.max(auc_scores):>12.4f}")
    
    # Per-class F1
    print(f"\n  Per-Class F1 (mean ± std across folds):")
    print("  " + "-" * 45)
    for c in range(num_classes):
        class_f1s = [m['per_class_f1'][c] for m in fold_metrics]
        print(f"    {label_names[c]:<12}: {np.mean(class_f1s):.4f} ± {np.std(class_f1s):.4f}")
    
    # Combined metrics
    print_subheader("Combined Validation Set Metrics")
    all_preds_combined = np.concatenate(all_val_preds)
    all_labels_combined = np.concatenate(all_val_labels)
    
    combined_thresholds = optimize_thresholds(all_preds_combined, all_labels_combined)
    combined_metrics = compute_metrics(all_labels_combined, all_preds_combined, combined_thresholds)
    
    print(f"  Macro F1:  {combined_metrics['macro_f1']:.4f}")
    print(f"  Macro AUC: {combined_metrics['macro_auc']:.4f}")
    print(f"\n  Optimal thresholds:")
    for i, name in enumerate(label_names):
        print(f"    {name:<10}: {combined_thresholds[i]:.3f}")
    
    # Save final results
    results_path = os.path.join(args.save_dir, 'cv_results.pt')
    torch.save({
        'fold_metrics': fold_metrics,
        'combined_metrics': combined_metrics,
        'combined_thresholds': combined_thresholds,
        'label_names': label_names,
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'auc_mean': np.mean(auc_scores),
        'auc_std': np.std(auc_scores),
        'training_time': cv_time,
        'args': vars(args)
    }, results_path)
    
    print(f"\n  💾 Results saved: {results_path}")
    print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return fold_metrics, combined_metrics


def main():
    parser = argparse.ArgumentParser(description='Train ECG Classifier with K-Fold CV')
    
    # Data
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory with preprocessed PTB-XL data')
    parser.add_argument('--num_samples', type=int, default=400,
                        help='Samples for synthetic data (if no data_dir)')
    
    # Model
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Attention heads')
    parser.add_argument('--downsample_factor', type=int, default=10, 
                        help='Sequence downsampling factor (5000->500 with factor=10)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs per fold')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Warmup epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    
    # Cross-validation
    parser.add_argument('--n_folds', type=int, default=10, help='Number of CV folds')
    
    # Checkpointing
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    train_cv(args)


if __name__ == '__main__':
    main()