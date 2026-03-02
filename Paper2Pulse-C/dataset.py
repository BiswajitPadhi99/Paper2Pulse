import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class PTBXLDataset(Dataset):
    def __init__(self, data_dir: str, normalize: bool = True, 
                 transform=None, indices: Optional[List[int]] = None):
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.transform = transform
        
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.label_names = self.metadata['label_names']
        self.num_classes = self.metadata['num_classes']
        self.sampling_rate = self.metadata.get('sampling_rate', 500)
        
        self.signal_mean = np.array(self.metadata.get('signal_mean_per_lead', [0.0]*12), dtype=np.float32)
        self.signal_std = np.array(self.metadata.get('signal_std_per_lead', [1.0]*12), dtype=np.float32)
        
        self.signals = np.load(self.data_dir / 'signals.npy', mmap_mode='r')
        self.labels = np.load(self.data_dir / 'labels.npy')
        
        self.indices = indices if indices is not None else list(range(len(self.signals)))
        self.class_frequencies = self.labels[self.indices].sum(axis=0) / len(self.indices)
        self.class_counts = self.labels[self.indices].sum(axis=0).astype(int)
        
        print(f"  [Dataset] Loaded {len(self.indices)} samples, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        actual_idx = self.indices[idx]
        
        # CRITICAL: Copy and replace NaN with 0
        signal = np.array(self.signals[actual_idx], dtype=np.float32)
        signal = np.nan_to_num(signal, nan=0.0)
        
        if self.normalize:
            signal = (signal - self.signal_mean) / (self.signal_std + 1e-8)
        
        if self.transform:
            signal = self.transform(signal)
        
        label = np.array(self.labels[actual_idx], dtype=np.float32)
        
        return {
            'signal': torch.from_numpy(signal),
            'label': torch.from_numpy(label)
        }
    
    def get_class_frequencies(self):
        return torch.from_numpy(self.class_frequencies.astype(np.float32))
    
    def get_class_counts(self):
        return self.class_counts
    
    def get_label_names(self):
        return self.label_names


class SyntheticECGDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=5, seq_length=5000, num_leads=12, seed=42):
        np.random.seed(seed)
        self.signals = np.random.randn(num_samples, seq_length, num_leads).astype(np.float32) * 0.3
        self.labels = np.zeros((num_samples, num_classes), dtype=np.float32)
        for i in range(num_samples):
            for c in range(num_classes):
                if np.random.random() < [0.4, 0.2, 0.15, 0.15, 0.1][c]:
                    self.labels[i, c] = 1
            if self.labels[i].sum() == 0:
                self.labels[i, 0] = 1
        self.num_classes = num_classes
        self.label_names = [f'Class_{i}' for i in range(num_classes)]
        self.class_frequencies = self.labels.mean(axis=0)
        self.class_counts = self.labels.sum(axis=0).astype(int)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return {'signal': torch.from_numpy(self.signals[idx]), 'label': torch.from_numpy(self.labels[idx])}
    
    def get_class_frequencies(self):
        return torch.from_numpy(self.class_frequencies.astype(np.float32))
    
    def get_class_counts(self):
        return self.class_counts
    
    def get_label_names(self):
        return self.label_names


def create_dataloaders(train_dataset, val_dataset, batch_size=32, num_workers=4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def load_ptbxl_dataset(data_dir, val_split=0.2, seed=42):
    full_dataset = PTBXLDataset(data_dir, normalize=True)
    n = len(full_dataset)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    val_size = int(n * val_split)
    train_dataset = PTBXLDataset(data_dir, normalize=True, indices=indices[val_size:].tolist())
    val_dataset = PTBXLDataset(data_dir, normalize=True, indices=indices[:val_size].tolist())
    return train_dataset, val_dataset
