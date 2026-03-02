"""
Loss Functions for Multi-Label ECG Classification

Includes:
- SimpleBCELoss: Robust BCE with class weighting (recommended to start)
- AdaptiveMultiLabelLoss: Combined loss with learnable weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SimpleBCELoss(nn.Module):
    """
    Simple BCE loss with class weighting.
    Most robust option - use this first to verify training works.
    """
    
    def __init__(self, class_frequencies: Optional[torch.Tensor] = None, 
                 pos_weight_factor: float = 1.0):
        super().__init__()
        
        if class_frequencies is not None:
            # pos_weight = (1 - freq) / freq for each class
            freq = torch.clamp(class_frequencies, min=0.05, max=0.95)
            pos_weight = (1 - freq) / freq * pos_weight_factor
            pos_weight = torch.clamp(pos_weight, min=0.5, max=10.0)  # Limit extreme weights
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Safety clamp
        logits = torch.clamp(logits, min=-20, max=20)
        
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        return loss, {'loss_total': loss.item()}


class AsymmetricLoss(nn.Module):
    """Asymmetric loss - harder on false negatives."""
    
    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0, 
                 margin: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.eps = eps
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, min=-20, max=20)
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1 - self.eps)
        
        # Positive samples
        pos_loss = targets * (1 - probs) ** self.gamma_pos * torch.log(probs)
        
        # Negative samples with margin
        probs_margin = (probs - self.margin).clamp(min=self.eps)
        neg_loss = (1 - targets) * probs_margin ** self.gamma_neg * torch.log(1 - probs_margin + self.eps)
        
        return (-pos_loss - neg_loss).mean()


class FocalLoss(nn.Module):
    """Focal loss - down-weights easy examples."""
    
    def __init__(self, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, min=-20, max=20)
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1 - self.eps)
        
        # Focal weights
        focal_pos = (1 - probs) ** self.gamma
        focal_neg = probs ** self.gamma
        
        # BCE components
        bce_pos = -targets * torch.log(probs)
        bce_neg = -(1 - targets) * torch.log(1 - probs)
        
        loss = focal_pos * bce_pos + focal_neg * bce_neg
        return loss.mean()


class DiceLoss(nn.Module):
    """Dice loss for F1 optimization."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        intersection = (probs * targets).sum(dim=0)
        union = probs.sum(dim=0) + targets.sum(dim=0)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class AdaptiveMultiLabelLoss(nn.Module):
    """
    Combined loss with learnable weights.
    Falls back to simple BCE if numerical issues occur.
    """
    
    def __init__(self, class_frequencies: Optional[torch.Tensor] = None,
                 gamma_pos: float = 1.0, gamma_neg: float = 4.0,
                 asym_margin: float = 0.05, focal_gamma: float = 2.0,
                 dice_smooth: float = 1.0):
        super().__init__()
        
        self.asym_loss = AsymmetricLoss(gamma_pos, gamma_neg, asym_margin)
        self.focal_loss = FocalLoss(focal_gamma)
        self.dice_loss = DiceLoss(dice_smooth)
        self.bce_fallback = SimpleBCELoss(class_frequencies)
        
        self.log_weights = nn.Parameter(torch.zeros(3))
        self.use_fallback = False
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Check for NaN inputs
        if torch.isnan(logits).any() or torch.isnan(targets).any():
            print("WARNING: NaN in inputs, using fallback BCE loss")
            logits = torch.nan_to_num(logits, nan=0.0)
            targets = torch.nan_to_num(targets, nan=0.0)
            return self.bce_fallback(logits, targets)
        
        try:
            l_asym = self.asym_loss(logits, targets)
            l_focal = self.focal_loss(logits, targets)
            l_dice = self.dice_loss(logits, targets)
            
            # Check for NaN losses
            if torch.isnan(l_asym) or torch.isnan(l_focal) or torch.isnan(l_dice):
                if not self.use_fallback:
                    print("WARNING: NaN loss detected, switching to fallback BCE")
                    self.use_fallback = True
                return self.bce_fallback(logits, targets)
            
            weights = F.softmax(self.log_weights, dim=0)
            total = weights[0] * l_asym + weights[1] * l_focal + weights[2] * l_dice
            
            if torch.isnan(total):
                return self.bce_fallback(logits, targets)
            
            return total, {
                'loss_total': total.item(),
                'loss_asym': l_asym.item(),
                'loss_focal': l_focal.item(),
                'loss_dice': l_dice.item(),
            }
        except Exception as e:
            print(f"WARNING: Loss computation error ({e}), using fallback")
            return self.bce_fallback(logits, targets)


if __name__ == "__main__":
    print("Testing loss functions...")
    
    torch.manual_seed(42)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 2, (8, 5)).float()
    class_freq = torch.tensor([0.4, 0.2, 0.15, 0.15, 0.1])
    
    # Test SimpleBCELoss
    criterion = SimpleBCELoss(class_freq)
    loss, details = criterion(logits, targets)
    print(f"SimpleBCELoss: {loss.item():.4f}")
    
    # Test AdaptiveMultiLabelLoss
    criterion2 = AdaptiveMultiLabelLoss(class_freq)
    loss2, details2 = criterion2(logits, targets)
    print(f"AdaptiveMultiLabelLoss: {loss2.item():.4f}")
    
    # Test with NaN
    logits_nan = torch.tensor([[float('nan'), 0.5, 0.3, 0.2, 0.1]])
    targets_nan = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])
    loss3, _ = criterion2(logits_nan, targets_nan)
    print(f"With NaN input (fallback): {loss3.item():.4f}")
    
    loss.backward()
    print("✓ All tests passed!")