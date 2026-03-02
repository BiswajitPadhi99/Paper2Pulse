"""
Training Script for ECG Signal Classifier (Single Train/Val Split)

Features:
- Verbose logging with timing
- Checkpoint saving every N epochs
- GPU monitoring
"""

import os
import sys
import argparse
import time
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np

from model import SignalClassifier, count_parameters
from losses import AdaptiveMultiLabelLoss
from dataset import PTBXLDataset, SyntheticECGDataset, create_dataloaders, load_ptbxl_dataset
from metrics import compute_metrics, optimize_thresholds, print_metrics


def print_header(text, char="=", width=80):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"{allocated:.2f}GB / {reserved:.2f}GB"
    return "N/A"


def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, criterion, optimizer, scheduler, device, 
                epoch, total_epochs, grad_clip=1.0):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    num_batches = len(loader)
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(loader):
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
        
        # Progress update
        if (batch_idx + 1) % max(1, num_batches // 5) == 0:
            progress = (batch_idx + 1) / num_batches * 100
            print(f"\r  Epoch {epoch+1}/{total_epochs} | "
                  f"Batch {batch_idx+1}/{num_batches} ({progress:.0f}%) | "
                  f"Loss: {loss.item():.4f}", end="", flush=True)
    
    print()
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    metrics = compute_metrics(labels, preds)
    metrics['loss'] = total_loss / len(loader)
    
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device, thresholds=None):
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


def train(args):
    print_header("ECG CLASSIFIER - SINGLE SPLIT TRAINING")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PyTorch: {torch.__version__}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    print_header("Loading Dataset")
    
    if args.data_dir:
        train_dataset, val_dataset = load_ptbxl_dataset(
            args.data_dir, val_split=args.val_split, seed=args.seed
        )
        num_classes = train_dataset.num_classes
        label_names = train_dataset.get_label_names()
        seq_length = 5000
        print(f"  Data directory: {args.data_dir}")
    else:
        print("  Using synthetic data...")
        from torch.utils.data import random_split
        dataset = SyntheticECGDataset(num_samples=args.num_samples, seed=args.seed)
        train_size = int(0.8 * len(dataset))
        train_dataset, val_dataset = random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        train_dataset.get_class_frequencies = dataset.get_class_frequencies
        train_dataset.get_class_counts = dataset.get_class_counts
        num_classes = dataset.num_classes
        label_names = dataset.label_names
        seq_length = 5000
    
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, args.batch_size, args.num_workers
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Num classes: {num_classes}")
    print(f"  Labels: {label_names}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Model
    print_header("Creating Model")
    model = SignalClassifier(
        num_classes=num_classes,
        d_model=args.d_model,
        max_seq_len=seq_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        downsample_factor=args.downsample_factor,
    ).to(device)
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = AdaptiveMultiLabelLoss(
        train_dataset.get_class_frequencies()
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)
    
    # Training
    print_header("Training")
    print(f"\n  {'Epoch':<8} {'Train Loss':<12} {'Train F1':<10} {'Val Loss':<12} {'Val F1':<10} {'Val AUC':<10} {'GPU Mem':<15}")
    print("  " + "-" * 85)
    
    best_f1 = 0
    best_state = None
    training_start = time.time()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            epoch, args.epochs
        )
        val_metrics, _, _ = validate(model, val_loader, criterion, device)
        
        gpu_mem = get_gpu_memory()
        improved = ""
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            best_state = deepcopy(model.state_dict())
            improved = " ⭐"
        
        print(f"  {epoch+1:<8} {train_metrics['loss']:<12.4f} {train_metrics['macro_f1']:<10.4f} "
              f"{val_metrics['loss']:<12.4f} {val_metrics['macro_f1']:<10.4f} {val_metrics['macro_auc']:<10.4f} "
              f"{gpu_mem:<15}{improved}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args)
            }, ckpt_path)
            print(f"  💾 Checkpoint saved: {ckpt_path}")
    
    training_time = time.time() - training_start
    print(f"\n  Training completed in {format_time(training_time)}")
    print(f"  Best validation F1: {best_f1:.4f}")
    
    # Load best and optimize thresholds
    if best_state:
        model.load_state_dict(best_state)
    
    print_header("Threshold Optimization")
    _, val_preds, val_labels = validate(model, val_loader, criterion, device)
    optimal_thresholds = optimize_thresholds(val_preds, val_labels)
    
    print(f"  Optimal thresholds:")
    for i, (name, thresh) in enumerate(zip(label_names, optimal_thresholds)):
        print(f"    {name:<10}: {thresh:.3f}")
    
    # Final evaluation
    final_metrics = compute_metrics(val_labels, val_preds, optimal_thresholds)
    print(f"\n  Final Macro F1:  {final_metrics['macro_f1']:.4f}")
    print(f"  Final Macro AUC: {final_metrics['macro_auc']:.4f}")
    
    print(f"\n  Per-class F1:")
    for i, name in enumerate(label_names):
        print(f"    {name:<10}: {final_metrics['per_class_f1'][i]:.4f}")
    
    # Save final model
    save_path = os.path.join(args.save_dir, 'best_model.pt')
    torch.save({
        'model_state_dict': best_state,
        'optimal_thresholds': optimal_thresholds,
        'final_metrics': final_metrics,
        'label_names': label_names,
        'args': vars(args)
    }, save_path)
    print(f"\n  💾 Best model saved: {save_path}")
    print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return model, optimal_thresholds


def main():
    parser = argparse.ArgumentParser(description='Train ECG Classifier')
    
    # Data
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--num_samples', type=int, default=400)
    
    # Model
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--downsample_factor', type=int, default=10,
                        help='Sequence downsampling factor (5000->500 with factor=10)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Checkpointing
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()